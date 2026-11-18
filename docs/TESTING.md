# How it was tested

Six layers, from the cheapest to the most realistic. The first four run on every commit and
need no network and no API key.

| Layer | What it answers | How to run it |
|---|---|---|
| 1. Unit and behaviour tests | Does each rule do what it says, and does the pipeline publish the right thing for a given document and a given model reply? | `make test` |
| 2. Replay of committed outputs | Are the committed sample and evaluation outputs exactly what the current code produces from the recorded model replies? | `make eval-replay` |
| 3. Scoring against hand labels | How do those outputs compare with what a person would record? | `uv run python -m changeorder.evaluation report` |
| 4. Stress test | What do the checks do when the model is wrong? | part of the report |
| 5. Independent review | What does someone who did not write the checks get past them? | pinned as tests, see below |
| 6. Live runs and manual checks | Does the real command work against a real model, and fail the way it should? | `make extract`, `make samples-run`, `make eval` |

## 1. The automated suite

970 tests, about 45 seconds. `make test`, or `uv run pytest`.

The model is never called. A scripted model (`ScriptedModel`) returns whatever reply the test
wrote, which is how a test plays a model that is right, wrong, fooled or broken and then checks
what the pipeline let through. The provider is `httpx`'s own mock transport, so every client
test goes through the real request, retry and error-mapping code. The PDFs are built inside the
tests with reportlab, so pypdf parses real files.

| File | Tests | What it covers |
|---|---|---|
| `tests/test_pipeline.py` | 117 | End to end with a scripted model: what is published, at which confidence, and whether review is asked for |
| `tests/test_false_accepts.py` | 94 | Wrong values a mistaken model could once get published, each beside the neighbouring case that must still be accepted (layer 5) |
| `tests/test_normalize.py` | 254 | The fixed rules: money, durations, dates, the day/month convention, direction words, labels, tick boxes, approval cues |
| `tests/test_hostile_input.py` | 34 | Text built to be slow, and input that used to end in a traceback |
| `tests/test_document.py` | 22 | Reading text and PDFs, page positions, tables stored column by column, and every refusal |
| `tests/test_llm.py` | 35 | The client: request shape, one bounded retry, the wall, structured-output fallback, error classes, no secrets in errors |
| `tests/test_extract.py` | 13 | Reading the model's reply: one correction, no third call, no made-up result |
| `tests/test_contract.py` | 38 | The published JSON Schema and the pydantic model give the same verdict on valid and broken documents |
| `tests/test_cli.py` | 22 | stdout, stderr and exit status; the program run as a subprocess; one run against a local HTTP endpoint with nothing mocked |
| `tests/test_prompt.py` | 7 | The prompt's pinned fingerprint, the strict-compatible schema, fencing, and text addressed to the model |
| `tests/test_config.py` | 11 | Settings: required variables, defaults, refusals, the local `.env` file |
| `tests/test_evaluation.py` | 28 | Scoring rules, the report's numbers, tamper detection, the stress test |
| `tests/test_samples.py` | 40 | Every committed output is schema-valid and is rebuilt byte for byte from its recorded run |
| `tests/vendor/` | 255 | The tests that travel with the three lifted Toolkit cores |

### Where each required case is tested

| Case from the brief | Tests |
|---|---|
| Correct extraction from straightforward input | `test_pipeline.py::test_a_straightforward_document_is_extracted_and_needs_no_review` |
| Missing fields and correct `null` behaviour | `test_fields_a_change_order_may_lack_are_null_without_review`, `test_a_missing_critical_field_is_null_and_sends_the_document_to_a_person`, `test_a_change_order_tied_to_no_contract_needs_review` |
| Explicit zero versus absent amount or schedule | `test_schedule_zero_is_stated_not_assumed`, `test_an_explicitly_zero_cost_change_is_zero`; `test_false_accepts.py::test_no_change_is_not_taken_on_the_models_word`, `test_no_change_the_document_states_about_the_field_is_zero`, `test_a_zero_in_a_table_is_read_under_its_own_column` |
| Positive and negative monetary changes | `test_the_sign_follows_the_document`, `test_a_sign_the_document_does_not_support_is_not_published`, `test_contract_sums_settle_the_sign_a_blank_form_leaves_open`; `test_false_accepts.py::test_the_sign_is_read_where_the_document_puts_it`, `test_a_sign_the_document_contradicts_is_refused` |
| Extension and reduction of schedule | `test_schedule_impact_is_read_in_days_with_its_sign`, `test_a_schedule_impact_that_cannot_be_read_one_way_is_not_published`; `test_false_accepts.py::test_a_duration_that_is_not_the_change_in_contract_time_is_refused` |
| Ambiguous dates and number formats | `test_dates_are_normalised_only_when_they_read_one_way`, `test_the_documents_other_dates_can_settle_the_order_of_day_and_month`, `test_a_model_cannot_settle_an_ambiguous_date_by_rewriting_it`, `test_number_formats_are_read_by_rule_or_refused`; `test_false_accepts.py::test_a_cost_code_does_not_settle_which_number_is_the_day`, `test_a_figure_that_is_not_the_whole_amount_is_not_read_as_one` |
| Competing monetary figures in one document | `test_contract_sums_that_differ_by_the_amount_corroborate_it`, `test_a_total_or_a_part_is_not_accepted_as_the_change`, `test_a_figure_the_contract_sums_contradict_is_not_published`, `test_a_figure_that_is_not_the_change_is_refused_by_its_label`, `test_what_was_requested_is_not_published_when_something_else_was_granted`; `test_false_accepts.py::test_a_contract_sum_is_never_the_change_whatever_arithmetic_is_offered`, `test_swapping_the_contract_sums_does_not_reverse_the_sign`, `test_a_figure_its_surroundings_do_not_make_the_change_is_refused` |
| Conflicting field evidence | `test_two_change_order_numbers_in_one_document_give_none`, `test_two_differing_candidates_under_the_same_kind_of_label_give_no_value`, `test_a_field_the_model_calls_unclear_is_null_and_keeps_its_note` |
| Approval-status ambiguity | `test_a_status_the_quoted_text_shows_is_accepted`, `test_an_approval_section_is_not_an_approval`, `test_an_approval_beside_a_rejection_gives_no_status`, `test_a_qualified_approval_is_approved_and_still_goes_to_a_person`; `test_false_accepts.py::test_approval_is_read_from_the_mark_not_from_the_word`, `test_one_signature_is_not_an_approval_while_another_line_is_unsigned` |
| Unsupported or hallucinated values | `test_a_value_the_document_does_not_support_is_never_published`, `test_a_real_value_under_another_fields_label_is_not_accepted`, `test_an_absence_the_document_contradicts_is_not_taken_on_the_models_word`; `test_false_accepts.py::test_a_value_the_model_assembled_from_pieces_is_not_published`, `test_a_model_cannot_scale_the_figure_itself` |
| Invalid model JSON and schema mismatches | `test_extract.py`: a wrong shape gets one correction; a second bad reply is an error; a cut-off reply is not retried. `test_hostile_input.py`: a reply nested thousands deep |
| Failed or truncated document extraction | `test_document.py`: unreadable files, scanned PDFs, encrypted PDFs, limits, a scanned page among text pages. `test_hostile_input.py`: a file that cannot be opened |
| LLM timeouts and provider failures | `test_llm.py`: timeouts, dropped connections, 401/402/403/404/429/5xx, a 200 that is not a completion; `test_cli.py`: each ends with exit status 4 and empty stdout |
| Review thresholds and business-rule failures | `test_two_caveats_put_a_value_under_the_review_threshold`, `test_a_quote_found_only_approximately_lowers_the_band`, `test_a_change_order_dated_before_its_contract_keeps_its_value_and_needs_review` |
| Document text written to manipulate the model | `test_prompt.py`: the fence a document cannot close, a document that addresses the model, values read from injected text, a model talked into "absent" |

Lint (`ruff`), formatting and strict type checking (`mypy --strict`) run with `make check`.

Line coverage, measured with `uv run --with pytest-cov pytest --cov=changeorder`: 96% over the
package. By module, the pipeline's own code is between 92% and 100%; the evaluation tool is at
80%, because its live-run path is not exercised by tests.

## 2. Committed outputs cannot drift

Every file in `samples/outputs/` and `eval/outputs/` has a twin in `diagnostics/` holding the
model's raw replies for that run. `make eval-replay`, and `tests/test_samples.py`, feed those
replies back through the current code and compare the result with the committed output byte
for byte, then regenerate `eval/REPORT.md` and compare that too. So:

- an output edited by hand fails the build;
- a change to the code that changes an output fails the build until the outputs are rebuilt
  (`python -m changeorder.evaluation rebuild <set>`), which shows up in the diff;
- a recording made under another prompt is refused, by its fingerprint.

`tests/test_evaluation.py` checks the check: it edits an output and expects the replay to fail.

## 3. Scoring against hand labels

34 documents: the six samples and 28 edge cases. Each was written and labelled by hand, and the
labels were committed before any model read the documents. Each field of each output is scored
as a correct value, a correct null, a wrong value, an invented value, or a missed value.
`eval/REPORT.md` has the results by set, by field, by confidence band, and for the review
decision; `docs/WRITEUP.md` section 7 says how to read them.

The documents are synthetic and were used to develop the checks, so these are development
results, not a held-out test.

## 4. The stress test

No model made a mistake that reached an output, so `src/changeorder/stress.py` makes them on
purpose: 770 single-field mistakes are injected into the recorded replies and run through the
deterministic stages, once with the label checks and once without. They are the wrong figure,
a reversed direction (alone, and with the contract sums swapped to match), a change reported
as no change, another date, the day of a date taken for a number of days, an approval read
from an approval section, a number taken out of a date, a name the model rewrote, and a value
reported absent. The report's last section has the table: 1 published with the checks, 126
without.

## 5. Independent review

The stress test is the author's own list of mistakes. To find the ones not on it, a second
reader (an AI model working in a separate session, with the code and no part in writing it)
was asked to get a wrong value published, and to make the pipeline slow or make it crash. It
wrote its own documents and model replies and reproduced each finding before reporting it.

The first round got wrong values published at 0.90 and 0.97 without review, in ten areas:
an explicit zero taken on the model's word, figures scaled by a word, a cost code that settled
a date's day/month order, contract sums used in the roles the model gave them, five ways of
getting a sign wrong, four ways of reading an approval that was not there, five durations that
were not the schedule impact, three figures that were not the change, a value published in the
model's own wording, and dates and numbers under the wrong label. It also found two patterns
that took between twelve seconds and over a minute on a long run of punctuation, and five
inputs that ended in a traceback. `docs/WRITEUP.md` section 7 lists them.

Each finding was fixed in the deterministic stages and is pinned:

- `tests/test_false_accepts.py`: every wrong value, as a refusal with its reason code, next to
  the neighbouring document that must still be accepted;
- `tests/test_hostile_input.py`: 26 shapes of slow text through the whole pipeline with a
  time limit, and the inputs that crashed. On the commit before the fix, three of those shapes
  took 18, 66 and 110 seconds on the development machine; they now take under one;
- the stress test, which now plays the same mistakes on all 34 labelled documents.

None of the fixes changed the prompt, and none changed a committed output.

## 6. Live runs and manual checks

Done by hand during development, on Windows with GNU make 4.4.1:

| Check | Result |
|---|---|
| `make extract FILE=samples/inputs/co-03-deductive-credit.pdf` against a live model | Exit 0; stdout is one JSON object that validates against the schema, in UTF-8; logs on stderr |
| The same command against a local OpenAI-compatible endpoint that serves the recorded reply | Exit 0; stdout byte-identical to the committed output. The same run without make is also a test (`test_cli.py`) |
| `make extract` with no `FILE` | Exit 2; stdout empty; usage on stderr |
| `make extract FILE=--help` | Nonzero (the program's 3: no such file); stdout empty. The name is passed after `--` |
| `make extract` with a missing file | Nonzero (the program's 3); stdout empty; "the document could not be read (not_found)" |
| `make extract` with no model configured, or with an endpoint that does not answer | Nonzero (the program's 4); stdout empty; the three variables named, or the timeout |
| The six samples, live, on `anthropic/claude-sonnet-5.5` | Recorded in `samples/diagnostics/` |
| The 28 edge cases, live, on `anthropic/claude-haiku-5.5` | Recorded in `eval/diagnostics/` |
| A provider that refuses the strict schema | Seen live: HTTP 400, then the same request completed in `json_object` mode. The schema was then made smaller and is accepted in strict mode |
| An account with no credit | Seen live: HTTP 402, reported as a failed run for each document, nothing written |
| The sample PDFs | Rendered to images and inspected; `samples/build_samples.py` rebuilds them byte for byte |

The live runs were made before the review fixes. Those fixes did not touch the prompt, the
client or the reader, and the recorded replies were replayed through the changed code; no new
live run was made afterwards.

## What was not tested

- **Real change orders.** Every document here is synthetic.
- **OCR.** There is none; scanned input is refused or reported.
- **Other providers.** The client is tested against a mock transport and a local endpoint for
  the OpenAI-compatible protocol in general, and was run live only through OpenRouter, with two
  models.
- **A second live run per document.** Each recorded run is a single run of a model that is not
  deterministic.
- **Load, concurrency and long documents.** One short document at a time.
- **Linux and macOS by hand.** Development was on Windows; the CI workflow runs the suite on
  Ubuntu.

## Try it yourself

None of this needs a key. Each command is one line and works the same in bash and PowerShell.

```bash
make setup
make test
make eval-replay
```

Run the pipeline on a sample, answering from the recorded reply of its live run:

```bash
uv run python -m changeorder samples/inputs/co-04-subcontractor-multi-amount.pdf --replay samples/diagnostics/co-04-subcontractor-multi-amount.json --diagnostics out/co-04.json
```

The JSON goes to stdout, the reasons for review to stderr, and `out/co-04.json` shows the
evidence for every field. To watch a check refuse something, change a figure in one of the
text samples and replay it: the recorded reply still points at the old figure, which is no
longer in the document, so the amount comes out null with `needs_review` true.

With a key in `.env` (see `.env.example`), run it live on any file:

```bash
make extract FILE=path/to/change-order.pdf
make explain FILE=path/to/change-order.pdf
```

Without make, the same things are `uv sync`, `uv run pytest`,
`uv run python -m changeorder.evaluation replay samples` (and `replay eval`), and
`uv run python -m changeorder -- path/to/change-order.pdf`.
