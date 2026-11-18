"""The command line's contract: one JSON object on stdout, everything else on stderr, and an
exit status that tells a processed document from a failure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from changeorder import cli
from changeorder.llm import LLMRateLimitedError, LLMTimeoutError, ScriptedModel
from changeorder.pipeline import contract_problems, process
from helpers import BASE, DATE_LINE, base_fields, model_reply, stated

GOOD = model_reply(**base_fields())


@pytest.fixture(autouse=True)
def no_local_settings(monkeypatch):
    """The developer's own .env and LLM_* variables never reach these tests."""
    monkeypatch.setattr(cli, "load_env_file", lambda *args: None)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def recorded(tmp_path: Path, text: str = BASE, reply: str = GOOD) -> tuple[Path, Path]:
    """A document, and the diagnostics of a run over it for `--replay` to answer from."""
    document = tmp_path / "change-order.txt"
    document.write_text(text, encoding="utf-8")
    diagnostics = tmp_path / "recorded.json"
    result = process(document, ScriptedModel([reply]))
    diagnostics.write_text(json.dumps(result.diagnostics()), encoding="utf-8")
    return document, diagnostics


def test_a_processed_document_prints_exactly_one_json_object_and_logs_elsewhere(tmp_path, capsys):
    document, diagnostics = recorded(tmp_path)

    status = cli.main([str(document), "--replay", str(diagnostics)])

    out, err = capsys.readouterr()
    assert status == 0
    data = json.loads(out)  # the whole of stdout is one JSON document
    assert out.startswith("{") and out.endswith("}\n")
    assert contract_problems(data) == []
    assert data["amount"] == {"value": 12450, "confidence": 0.9} and data["needs_review"] is False
    assert "needs_review=false" in err and "read change-order.txt" in err
    assert "INFO" not in out and "needs_review=" not in out


def test_a_document_that_needs_review_is_still_a_success(tmp_path, capsys):
    ambiguous = {**base_fields(), "date": stated("03/04/2026", "Date: 03/04/2026")}
    document, diagnostics = recorded(
        tmp_path, BASE.replace(DATE_LINE, "Date: 03/04/2026\n"), model_reply(**ambiguous)
    )

    status = cli.main([str(document), "--replay", str(diagnostics)])

    out, err = capsys.readouterr()
    assert status == 0
    assert json.loads(out)["needs_review"] is True
    assert json.loads(out)["date"] == {"value": None, "confidence": 0.0}
    assert "WARNING needs_review=true: date: day_and_month_ambiguous" in err


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("missing.pdf", None, "not_found"),
        ("photo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "unsupported_type"),
        ("empty.txt", b"", "no_text"),
        ("broken.pdf", b"%PDF-1.7 not a pdf", "malformed"),
    ],
)
def test_an_unreadable_document_fails_with_status_3_and_prints_nothing(
    tmp_path, capsys, name, content, code
):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)

    status = cli.main([str(path), "--replay", str(tmp_path / "unused.json")])

    out, err = capsys.readouterr()
    assert status == cli.EXIT_DOCUMENT == 3
    assert out == ""  # never successful-looking JSON
    assert f"ERROR the document could not be read ({code})" in err


def test_a_missing_model_configuration_fails_with_status_4_and_names_what_to_set(tmp_path, capsys):
    document, _ = recorded(tmp_path)

    status = cli.main([str(document)])

    out, err = capsys.readouterr()
    assert status == cli.EXIT_MODEL == 4 and out == ""
    assert "LLM_API_KEY, LLM_BASE_URL, LLM_MODEL must be set" in err


@pytest.mark.parametrize(
    "replies",
    [
        [LLMTimeoutError("the model call did not finish within 120 seconds")],
        [LLMRateLimitedError("the model service is rate-limited or overloaded (HTTP 429)")],
        ["not json at all", '{"still": "the wrong shape"}'],
    ],
)
def test_a_failed_model_call_fails_with_status_4_and_prints_nothing(
    tmp_path, capsys, monkeypatch, replies
):
    document, _ = recorded(tmp_path)
    monkeypatch.setattr(cli, "_model", lambda replay: ScriptedModel(replies))

    status = cli.main([str(document)])

    out, err = capsys.readouterr()
    assert status == 4 and out == ""
    assert "ERROR the model gave no usable answer" in err


def test_a_recording_that_does_not_fit_the_request_is_a_failure(tmp_path, capsys):
    document, diagnostics = recorded(tmp_path)
    diagnostics.write_text('{"run": {"replies": []}}', encoding="utf-8")  # nothing recorded
    assert cli.main([str(document), "--replay", str(diagnostics)]) == 4

    diagnostics.write_text("not a diagnostics file", encoding="utf-8")
    assert cli.main([str(document), "--replay", str(diagnostics)]) == 4
    assert capsys.readouterr().out == ""


def test_a_missing_argument_is_a_usage_error():
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code == cli.EXIT_USAGE == 2


def test_diagnostics_go_to_their_own_file_and_leave_stdout_alone(tmp_path, capsys):
    document, diagnostics = recorded(tmp_path)
    target = tmp_path / "out" / "explain.json"

    status = cli.main(
        [str(document), "--replay", str(diagnostics), "--diagnostics", str(target), "--quiet"]
    )

    out, err = capsys.readouterr()
    assert status == 0 and err == ""  # --quiet: nothing to say about a clean run
    assert set(json.loads(out)) == {*base_fields(), "needs_review"}  # the contract and nothing more
    explained = json.loads(target.read_text(encoding="utf-8"))
    assert explained["run"]["prompt"] == "change_order_facts@1"
    assert len(explained["run"]["prompt_fingerprint"]) == 64
    assert explained["run"]["replies"][0]["text"] == GOOD
    amount = explained["fields"]["amount"]
    assert amount["state"] == "accepted" and amount["band"] == "supported"
    assert amount["as_written"] == "$12,450.00"
    assert amount["evidence"][0]["kind"] == "exact" and amount["evidence"][0]["line"] == 10
    assert explained["document"]["sha256"] and explained["findings"] == []


def test_as_a_program_it_writes_utf8_json_to_stdout_and_logs_to_stderr(tmp_path):
    text = BASE.replace("Maple Court Apartments", "Maple Court Apartments – Phase II")
    fields = {
        **base_fields(),
        "project_name": stated(
            "Maple Court Apartments – Phase II", "Project: Maple Court Apartments – Phase II"
        ),
    }
    document, diagnostics = recorded(tmp_path, text, model_reply(**fields))
    command = [sys.executable, "-m", "changeorder"]

    done = subprocess.run(
        [*command, str(document), "--replay", str(diagnostics)], capture_output=True, check=False
    )

    assert done.returncode == 0
    data = json.loads(done.stdout.decode("utf-8"))  # UTF-8 whatever the console's code page
    assert data["project_name"]["value"] == "Maple Court Apartments – Phase II"
    assert (
        b"needs_review=false" in done.stderr
        and b"needs_review" not in done.stdout.split(b'"needs_review"')[0]
    )

    failed = subprocess.run(
        [*command, str(tmp_path / "nowhere.pdf")], capture_output=True, check=False
    )
    assert failed.returncode == 3 and failed.stdout == b""
    assert b"could not be read" in failed.stderr
