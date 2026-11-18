"""The committed sample and evaluation outputs are what the pipeline produces.

Every output in samples/outputs and eval/outputs came from a live model run whose raw replies
are recorded beside it in diagnostics/. These tests rebuild each output from those replies with
the current code and compare byte for byte, so an output cannot be edited by hand and the code
cannot change underneath it without a failure here. No network and no key are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changeorder.evaluation import REPORT, build_report, replay
from changeorder.pipeline import contract_problems
from changeorder.prompt import fingerprint

REPO = Path(__file__).resolve().parents[1]
SAMPLES = sorted((REPO / "samples" / "inputs").iterdir())
OUTPUTS = sorted(
    [*(REPO / "samples" / "outputs").glob("*.json"), *(REPO / "eval" / "outputs").glob("*.json")]
)


def test_there_are_six_samples_each_with_a_label_an_output_and_a_recorded_run():
    assert len(SAMPLES) == 6
    assert {path.suffix for path in SAMPLES} == {".pdf", ".txt"}
    for sample in SAMPLES:
        for folder in ("truth", "outputs", "diagnostics"):
            assert (REPO / "samples" / folder / f"{sample.stem}.json").is_file(), (
                folder,
                sample.name,
            )


@pytest.mark.parametrize(
    "output", OUTPUTS, ids=lambda path: f"{path.parent.parent.name}/{path.stem}"
)
def test_every_committed_output_satisfies_the_published_schema(output):
    data = json.loads(output.read_text(encoding="utf-8"))

    assert contract_problems(data) == []
    assert list(data)[-1] == "needs_review" and len(data) == 10


@pytest.mark.parametrize("name", ["samples", "eval"])
def test_committed_outputs_are_rebuilt_exactly_from_their_recorded_replies(name):
    assert replay(REPO / name) == 0


@pytest.mark.parametrize("name", ["samples", "eval"])
def test_every_recorded_run_names_a_live_model_and_the_current_prompt(name):
    for recorded in sorted((REPO / name / "diagnostics").glob("*.json")):
        run = json.loads(recorded.read_text(encoding="utf-8"))["run"]
        assert run["prompt_fingerprint"] == fingerprint(), recorded.name
        assert "/" in run["model"]["model"] and "scripted" not in run["model"], recorded.name
        assert run["replies"] and all(reply["text"] for reply in run["replies"]), recorded.name


def test_the_committed_report_is_what_the_committed_outputs_produce(monkeypatch):
    monkeypatch.chdir(REPO)

    report, _ = build_report()

    assert REPORT.read_text(encoding="utf-8") == report
