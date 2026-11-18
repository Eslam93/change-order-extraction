# Change order extraction

Turns one construction change order (a PDF with a text layer, or pasted text) into one
validated JSON object: nine fields, each with a value and a confidence, plus `needs_review`.

The rule the code follows: **trust the document, verify the extraction, and abstain when the
evidence is not there.** A language model reads the document and points at where each fact is
stated. It never produces a final value. Code finds every quote in the document, reads numbers
and dates from the document's own characters, checks that the label beside a value fits the
field, and publishes null with a recorded reason whenever any of that fails.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and GNU make. Python 3.12 is installed by uv.

```bash
make setup                      # uv sync
cp .env.example .env            # then set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
make extract FILE=samples/inputs/co-03-deductive-credit.pdf
```

Any OpenAI-compatible chat-completions endpoint works. Nothing about a provider or a model is
in the code.

No key at hand? Every committed run keeps the model's raw replies, and the pipeline can answer
from them instead of calling a model:

```bash
uv run python -m changeorder samples/inputs/co-03-deductive-credit.pdf \
    --replay samples/diagnostics/co-03-deductive-credit.json
```

Either way, stdout carries exactly one JSON object and nothing else (shown here with each
field on one line and the description cut short):

```json
{
  "project_name": { "value": "Riverbend Logistics Center – Building C", "confidence": 0.9 },
  "contract_number": { "value": "CIP-RLC-0932", "confidence": 0.9 },
  "change_order_number": { "value": "CO-011", "confidence": 0.9 },
  "date": { "value": "2026-01-12", "confidence": 0.9 },
  "requested_by": { "value": "Calloway Industrial Properties (Owner)", "confidence": 0.9 },
  "description": { "value": "Delete the mezzanine office build-out (Rooms M-101 through M-108) ...", "confidence": 0.9 },
  "amount": { "value": -110983.75, "confidence": 0.97 },
  "schedule_impact_days": { "value": -10, "confidence": 0.97 },
  "approval_status": { "value": "approved", "confidence": 0.75 },
  "needs_review": false
}
```

Logs go to stderr. The exit status separates a processed document from a failure:

| Status | Meaning | stdout |
|---|---|---|
| 0 | The document was processed. `needs_review` may be true. | one JSON object |
| 2 | Usage error (no `FILE`) | empty |
| 3 | The document could not be read: not found, not a PDF or text, encrypted, damaged, scanned with no text layer, garbled, or over a limit | empty |
| 4 | The model is not configured, could not be reached, timed out, or gave no usable answer after one correction | empty |
| 5 | The result would break the published schema (a bug) | empty |

A caught error is never turned into a result.

To see why each value was accepted or refused, `make explain FILE=<path>` also writes
`out/diagnostics.json`: for every field the quotes, where they were found, the caveats and the
reasons, and for the document the findings that asked for review.

## The output

The contract is [`src/changeorder/change_order.schema.json`](src/changeorder/change_order.schema.json)
(JSON Schema, Draft 2020-12). The pipeline checks the exact object it is about to print against
that file.

- All nine fields are always present, each with exactly `value` and `confidence`.
- A value the document does not state is `null` with confidence `0.0`. It is never `0` and
  never `""`. `0` means the document says so: "no change to the Contract Time" is `0`; a
  document that never mentions schedule is `null`.
- `amount` is the signed change to the contract value: negative for a credit. It is never the
  original or the revised contract total. `schedule_impact_days` is signed the same way.
- `date` is `YYYY-MM-DD`. A date such as `03/04/2026` is `null` unless another date in the
  same document proves which number is the day.
- `approval_status` is one of `approved`, `rejected`, `pending`, `draft`, `void`, or `null`
  when the document does not establish a state. An approval section, empty signature lines and
  "not valid until signed" are not an approval.

## How it works

```text
file ──▶ document.py   read the text layer (layout mode) or the text file; refuse what cannot be read
     ──▶ extract.py    one model call: for each field, the value as written, quotes, and
                       "stated" / "absent" / "unclear" (prompt.py, llm.py)
     ──▶ fields.py     per field: find the quotes (evidence.py), find the value inside the quoted
                       lines, read it by rule (normalize.py), check its label, its sign and the
                       contract sums; accept, or record why not
     ──▶ validate.py   across the document: two change order numbers, approval against rejection,
                       unread pages, text addressed to the model, values the model called absent
     ──▶ confidence.py a band per accepted value; needs_review from the findings
     ──▶ pipeline.py   check the result against the published schema; print it (cli.py)
```

A value is published only when the model says the document states it, a quote for it is found
in the document, the value is found inside the quoted lines and read from them by a fixed rule,
and nothing contradicts it. Otherwise the field is `null` and the reason is recorded.

## Confidence and review

Confidence is a band, not a probability, and the model is never asked for it.

| Value | Band | What the checks established |
|---|---|---|
| 0.97 | corroborated | Verbatim evidence, read by rule, and an independent check agrees: the contract sums before and after differ by exactly this amount, or a duration is written in words and in figures that match |
| 0.90 | supported | Verbatim evidence, read by rule, a label that fits, nothing competing or contradicting |
| 0.75 | qualified | One caveat: the quote was found only approximately, the day/month order came from the document's other dates, weeks were converted to days, approval rests on a signature alone, or part of the document could not be read |
| 0.50 | weak | Two or more caveats |
| 0.00 | no value | `null` |

`needs_review` is true when any of these holds: a field could not be resolved; the document
contradicts itself; a critical field (`change_order_number`, `amount`, `approval_status`) is
null, or neither a project name nor a contract number ties the change order to a contract; a
page could not be read; the document contains text addressed to a model; the model called a
field absent that the document seems to hold; an accepted value is below 0.70; or a rule finds
a value suspicious (a change order dated before its contract, a qualified approval).

A missing `date`, `requested_by`, `description` or `schedule_impact_days` does not by itself
ask for review: many change orders do not state them.

## Results

[`eval/REPORT.md`](eval/REPORT.md) is generated from the committed outputs and labels. In
short, on documents written for this project:

| | Six samples | 28 edge cases |
|---|---|---|
| Model | `anthropic/claude-sonnet-5.5` | `anthropic/claude-haiku-5.5` |
| Fields correct (value or null) | 53 of 54 | 247 of 252 |
| Published values that are correct | 47 of 47 | 231 of 231 |
| Wrong or invented values | 0 | 0 |
| Stated values left null (abstentions) | 1 | 5 |
| Documents flagged for review | 3 of 6, all warranted | 12 of 28: 8 warranted, 4 not |
| Documents that needed review and were passed | 0 | 0 |

No model made a mistake that reached an output, so a stress test makes them deliberately: 551
single-field mistakes injected into the recorded replies. One is published (a bare "Owner"
taken from running text as the project name, in a document already flagged). With the label
checks switched off, 107 are.

These numbers describe synthetic documents written by the pipeline's author and used to
develop it. They are not a held-out test, and the confidence values are not calibrated. The
write-up says what they do and do not show.

## Repository

```text
src/changeorder/     the pipeline (start with pipeline.py, then fields.py)
  vendor/            three cores lifted from the Toolkit, with a README
tests/               480 tests of this code, 255 that travel with the lifted cores
samples/             six documents, their labels, the delivered outputs, the recorded runs
eval/                28 labelled edge cases, their outputs, REPORT.md
docs/WRITEUP.md      design, decisions, limits, what production would change
```

| Command | Does |
|---|---|
| `make extract FILE=<path>` | Extract one change order: JSON on stdout, logs on stderr |
| `make explain FILE=<path>` | The same, and write `out/diagnostics.json` |
| `make test` | The test suite: no network, no key |
| `make check` | Lint, types and tests |
| `make eval-replay` | Rebuild every committed output from its recorded replies and compare |
| `make samples-run`, `make eval` | Run the samples or the edge cases against the live model |
| `make samples` | Rebuild the sample PDFs from `samples/build_samples.py` |

## Limits worth knowing before relying on it

- **No OCR.** A scanned PDF is refused. A scanned page inside a text PDF is reported, lowers
  every confidence one band and forces review; what is on it is not read.
- **English only.** The label and approval lexicons are English, and deliberately narrow: an
  unfamiliar phrasing is refused, not guessed. That costs recall (see the abstentions above).
- **Amounts written only in words, three-decimal currencies and tables without a header row**
  are refused.
- **One record per document.** A file holding several change orders is not split.
- **Evaluated on synthetic documents only.**

[`docs/WRITEUP.md`](docs/WRITEUP.md) covers the design decisions, the confidence method and its
limits, the evaluation, the failure modes and what a production deployment would change.
