# Code lifted from the Toolkit

Three small cores copied from `github.com/Eslam93/ToolKit` at commit `bc4f149` (2026-11-06).
The Toolkit is built to be lifted from: each core is independent and travels with its tests.
The copies are here so this project installs and runs with no access to that repository.

| Here | From the Toolkit | Used for | Changed |
|---|---|---|---|
| `quote_matching/` | `document-extraction/toolkit/quote_matching/core/` | Finding a model's quote in the document's text: exact, then after folding case, typographic quotes and dashes, ligatures and white space, then nearly word for word, each with a span in the original text (`evidence.py`) | Nothing: byte-identical |
| `number_reading.py` | `llm-agent-core/toolkit/number_text/core/reading.py` | One decimal rule for figures as people write them (`1,234.56`, `1.234,56`, `12,50`), and the document's own decimal mark (`normalize.py`) | Nothing: byte-identical |
| `injection_sanitizer.py` | `document-extraction/toolkit/injection_sanitizer/core/sanitizer.py` | Removing invisible characters before any position is taken, finding text written as an instruction to a model, and fencing the document with a random token (`document.py`, `validate.py`, `prompt.py`) | Nothing: byte-identical |

Their tests are in `tests/vendor/`, 255 of them, with only the import line changed. One test was
left behind: the sanitizer's ablation drill, which needs a Toolkit core that was not lifted.

These files are excluded from `ruff format` and held to looser lint rules in `pyproject.toml`,
so that they can still be compared with their source.

## Adapted, not copied

Several modules follow a Toolkit core's design and were rewritten smaller for this job. The
Toolkit versions are asynchronous, built for many documents and many model calls; this
pipeline makes one synchronous call per document.

| Here | Follows | What was kept | What was dropped |
|---|---|---|---|
| `llm.py` | `model_provider`, `walled_http_call`, `classified_retry` | One wall for the whole call, one retry only for failures a retry can fix, named error classes, no response text or key in any error | Async, provider pinning, reasoning settings, tool calls, structlog |
| `extract.py` | `structured_output` | A strict schema on the request, validation here regardless, one corrective retry that names the problem, `first_json_object` | The fallback value: a failed extraction raises instead |
| `prompt.py` | `versioned_prompts` | An id, a version and a fingerprint over prompt and schema, pinned by a test | The catalogue and worked examples |
| `document.py` | `document_text` | Detection by first bytes, page and size caps, one error class with stable codes | DOCX; plain extraction as the default (layout mode is used here) |
| `llm.ScriptedModel` | `mock_model` | A model that answers from a script and records what it was shown | Tool calls and responders |
| `evaluation.py` | `field_evals` | Scoring each field after normalising by its kind | Replaced the rest: outcomes that separate wrong values from abstentions, confidence bands, review, replay |

Not used: `document_blocks`, `text_chunks`, `grounded_records`, `record_extraction`,
`extraction_pipeline` and `fan_out_jobs`. They chunk long documents and merge many records
across them. A change order is one short document and one record.
