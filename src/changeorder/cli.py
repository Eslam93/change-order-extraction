"""The command line: one file in, exactly one JSON object on stdout, everything else on stderr.

    changeorder FILE [--diagnostics PATH] [--replay PATH]

Exit status: 0 when the document was processed (whether or not it needs review); 2 for a usage
error; 3 when the document could not be read; 4 when the model could not be reached, was not
configured, or gave no usable answer; 5 when the result would break the published schema (a
bug). On any failure stdout stays empty: a caught error is never turned into a result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, llm_settings, load_env_file
from .document import DocumentError, read_document
from .llm import ChatModel, LLMError, OpenAICompatibleModel, Reply, ScriptedModel
from .pipeline import ContractError, Result, process_document

EXIT_OK, EXIT_USAGE, EXIT_DOCUMENT, EXIT_MODEL, EXIT_INTERNAL = 0, 2, 3, 4, 5

log = logging.getLogger("changeorder")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="changeorder",
        description="Extract one construction change order (text-layer PDF or plain text) into "
        "validated JSON on stdout. Logs go to stderr.",
    )
    parser.add_argument(
        "file", type=Path, help="the change order: a PDF with a text layer, or a text file"
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        metavar="PATH",
        help="also write evidence, findings and run details to this JSON file",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        metavar="PATH",
        help="answer from the model replies recorded in a diagnostics file: no network, no key",
    )
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env_file()

    try:
        # The document first: a file that cannot be read is the more useful thing to report,
        # and no model is configured or called for it.
        document = read_document(args.file)
        result = process_document(document, _model(args.replay))
    except DocumentError as exc:
        return _fail(EXIT_DOCUMENT, f"the document could not be read ({exc.code}): {exc.message}")
    except ConfigError as exc:
        return _fail(EXIT_MODEL, f"the model is not configured: {exc}")
    except LLMError as exc:
        return _fail(EXIT_MODEL, f"the model gave no usable answer: {exc}")
    except ContractError as exc:
        return _fail(EXIT_INTERNAL, str(exc))

    _report(result)
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(_dump(result.diagnostics()), encoding="utf-8", newline="\n")
        log.info("diagnostics written to %s", args.diagnostics)
    _emit(_dump(result.output.to_dict()))
    return EXIT_OK


def _model(replay: Path | None) -> ChatModel:
    if replay is None:
        return OpenAICompatibleModel(llm_settings())
    try:
        recorded = json.loads(replay.read_text(encoding="utf-8"))["run"]
        replies = [Reply.from_dict(item) for item in recorded["replies"]]
    except (OSError, ValueError, KeyError, TypeError):
        raise ConfigError(f"{replay} is not a diagnostics file with recorded replies") from None
    name = recorded.get("model", {}).get("model", "unknown")
    return ScriptedModel(replies, name=f"replay of {name}")


def _report(result: Result) -> None:
    document = result.document
    log.info(
        "read %s: %s, %d page(s), %d characters",
        document.name,
        document.kind,
        document.page_count,
        len(document.text),
    )
    last = result.run.replies[-1]
    log.info(
        "model %s answered (%s; %d prompt + %d completion tokens%s)",
        last.model or result.model.get("model"),
        f"response format {last.response_format}",
        sum(reply.prompt_tokens for reply in result.run.replies),
        sum(reply.completion_tokens for reply in result.run.replies),
        ", after one corrective retry" if result.run.corrected else "",
    )
    if result.output.needs_review:
        log.warning("needs_review=true: %s", "; ".join(result.review_reasons))
    else:
        log.info("needs_review=false")


def _dump(data: object) -> str:
    # allow_nan=False: NaN and Infinity are not JSON and must never reach stdout as if they were.
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _emit(text: str) -> None:
    """Writes UTF-8 to stdout whatever the console's code page is: JSON is UTF-8."""
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
    else:
        buffer.write(text.encode("utf-8"))
    sys.stdout.flush()


def _fail(status: int, message: str) -> int:
    log.error(message)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
