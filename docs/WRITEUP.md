# Technical write-up

How the change order extraction pipeline works, why it is built this way, what was measured,
and where it stops. Every number here comes from `eval/REPORT.md`, which is generated from
committed files; every behaviour described is covered by a test.

## 1. Architecture and data flow

One document goes through six stages. Each is one module, and each arrow is one function call
in `pipeline.py`.

| Stage | Module | In | Out |
|---|---|---|---|
| Read | `document.py` | a file | canonical text with page spans, or `DocumentError` |
| Ask | `prompt.py`, `llm.py`, `extract.py` | the text | candidates: per field a status, the value as written, quotes |
| Decide | `fields.py` with `evidence.py`, `normalize.py` | candidates and text | per field: accepted with a typed value, absent, or unresolved with a reason |
| Validate | `validate.py` | decisions and text | findings; conflicts turn their field unresolved |
| Score | `confidence.py` | decisions and findings | a confidence band per value, `needs_review` |
| Deliver | `pipeline.py`, `cli.py` | everything | the contract JSON, checked against the published schema |

Two shapes are kept apart on purpose (`models.py`). `ChangeOrder` is the public contract: nine
fields, each exactly `{value, confidence}`, plus `needs_review`. `Decision` and `Finding` hold
the evidence, caveats and review reasons. They never enter the public JSON; `--diagnostics`
writes them to a separate file, together with the model's raw replies and the run's settings.

The canonical text is the hinge. It is what the model is shown, what quotes are searched in,
and what every recorded position refers to, so "the model quoted line 14 of page 1" is a
checkable statement.

## 2. Why this approach

**The model points; code decides.** A model is good at what rules are bad at: telling which of
nine figures on a page is the change itself, which of nine dates belongs to the change order,
whether a signature block was signed. It is unreliable at what rules are good at: copying a
number exactly, keeping a sign, not filling a gap with something plausible. So the model is
asked only for the first kind of work. For each field it returns the value *as written*,
quotes that contain it, and one of `stated`, `absent`, `unclear`. It is told that code will
check every quote and discard what it cannot find, and that "absent" is a good answer.

Everything published is then derived by code from text that provably exists in the document.
A model cannot invent a number into the output, because no number in the output comes from the
model: `-110983.75` is parsed from the characters `($110,983.75)` on a line the model pointed
at. Its rendering of a value is used only to pick the token.

**One call, no orchestration.** A change order is one or two pages. There is no chunking, no
retrieval, no second pass and no agent. The call carries a strict JSON schema where the
provider supports one. A reply that does not parse gets exactly one corrective attempt that
names what was wrong with its shape; a reply cut off at the output limit is not retried,
because the same request would be cut again.

**Structured output degrades in the open.** With `LLM_STRUCTURED_OUTPUT=auto` the request
first carries a strict `json_schema` response format. A provider that rejects the request
(HTTP 400, 404, 415, 422, 501) is asked again with `json_object`, then with no response format;
the prompt always describes the JSON, so the reply can be read in any mode. Each step down is
logged, the mode that produced the reply is recorded with the run, and the reply is validated
by code in every mode. This was exercised for real during development: the first schema, with
every shared shape written out in place, was refused by the provider ("the compiled grammar is
too large") and the run completed in `json_object` mode. The schema now keeps its `$ref`s and is
accepted in strict mode.

**What was rejected.** Rules alone cannot read layouts they have not seen. A model alone
cannot be trusted with numbers, and its self-reported confidence means nothing. A second model
as a judge would add cost and another unverified opinion; the checks here are deterministic, so
they can be tested, counted and replayed.

## 3. How source evidence is verified

Finding a quote proves the text exists. It does not prove the text supports the value. The
brief's example is the standard one: "Revised Contract Sum: $850,000" is a real quote and does
not support an amount of 850000. Verification therefore has four layers, applied per field in
`fields.py`.

**1. The quote is in the document** (`evidence.py`). Each quote is searched in three tiers, and
the tier is recorded: `exact`; `normalized`, after folding case, typographic quotes and dashes,
ligatures and white space, so a quote that joins two lines still matches; `fuzzy`, when at
least 85% of its words appear in order within a stretch they fill to 75%. The fuzzy tier is
strict on purpose: it forgives a dropped word, not a paraphrase. It also lowers the value's
confidence band.

**2. The value is in the quoted lines, and is read from them** (`normalize.py`). For a name or
an identifier, the value must be found inside the lines the quotes touch, and what is published
is sliced from the document, so the model's change of case or dashes never reaches the output.
For a number, a duration or a date, the scanners find the tokens on the quoted lines, the
model's value selects one by what it reads to, and the token is read by a fixed rule that
refuses what it cannot settle. A description is rebuilt sentence by sentence from the
document's own wording, entirely or not at all.

**3. The label fits the field.** The words beside a value say what it is. A figure labelled
"Revised Contract Sum", "Subtotal" or "not to exceed", a figure followed by a unit or preceded
by "@", a date labelled "Contract Date" or sitting in a signature block, a duration in "valid
for 15 days", an identifier under "Contract No." offered as the change order number: each is
refused with a reason. A date is accepted only under a label that makes it the document's own
("Date:", "Issued:", "Date Prepared:") or alone at the head of its line, as a letter's date is.
In a table row, the label is the column's heading in the header row above.

**4. Nothing contradicts it.**
- *Sign.* A figure written as negative (parentheses, a minus sign, CR) or wording that goes one
  way only ("deduct", "credit", "add") must agree with the direction the model reported. A
  decrease that nothing in the document shows is not accepted on the model's word. A blank
  form that prints both "(increased) (decreased)" settles nothing by itself.
- *Arithmetic.* The model also points at the contract sum before and after the change. If they
  differ by exactly the amount, the amount is corroborated and its sign is confirmed. If they
  differ by something else, the amount is refused. If only the original sum is known, a
  difference neither corroborates nor contradicts, because earlier change orders the document
  does not list could explain it.
- *Approval.* A status is accepted only if the quoted lines show it: a statement, a stamp, a
  ticked box, or a completed signature. The word does not count when it is an unticked option,
  is negated, sits in a condition ("not valid until signed", "must be approved by") or labels a
  blank ("Approved by Owner: ______").

The stress test in section 7 measures what layers 3 and 4 are worth: with the label checks
switched off, 107 of 551 injected mistakes are published; with them on, 1.

**Untrusted text.** Invisible characters are removed before any position is taken. The
document reaches the model fenced between markers carrying a random token it cannot guess, and
the prompt says the fenced text is data. Text written as an instruction to a model is found
deterministically (`vendor/injection_sanitizer.py`): the document is then always sent to
review, every confidence drops a band, and a value read out of the instruction's own sentence
is refused, even though its quote is real. A model talked into reporting a field absent is
caught by the absence checks in section 4.

## 4. Normalisation and business validation

### Normalisation rules

| Field | Rule | Refused |
|---|---|---|
| `amount` | One decimal rule (lifted from the Toolkit): with a comma and a point, the last is the decimal mark; one mark before two digits is decimal; one mark before three digits groups thousands, unless the document's own figures prove the other mark. Parentheses, a minus sign or CR make it negative. Kept in `Decimal`; emitted as a JSON number. | Groups that are not thousands; three decimals where the document proves a decimal mark; amounts written only in words; several figures in one value |
| `schedule_impact_days` | Whole days. Weeks are multiplied by seven, with a caveat. Working days are not converted to calendar days: the number is taken as stated. A duration in words and figures must agree. | Months; fractions of a day; "ten (12) days" |
| `date` | ISO, month-name and numeric forms. A numeric date whose first or second number exceeds 12 reads one way. Otherwise it needs the document's convention, taken from its other numeric dates when they prove one and do not disagree; the value then carries a caveat. Two-digit years are read as 20xx below 80, with a caveat. | `03/04/2026` with nothing to settle it; impossible dates |
| `approval_status` | `approved`, `rejected`, `pending`, `draft`, `void`, by the cues in section 3. A qualified approval ("approved as noted") is `approved`, one band lower, and always reviewed, because the output cannot carry the condition. | Any status the quoted text does not show |
| Names, identifiers | The document's characters, white space collapsed. | A value interrupted by another column is kept with a caveat; one that is not in its quote is refused |

**Missing versus zero.** `0` needs evidence like any other value: a zero figure, or wording
that says nothing changes ("no change to the Contract Time", "at no cost to the Owner",
"none"). "N/A" is read as no change one band lower. "TBD", "to follow" and reserved rights are
not zero: the model reports `unclear`, the field is null and the document is reviewed. A field
the document never mentions is null, at confidence 0.0, and for the non-critical fields that
needs no review.

**No currency conversion.** The amount is the figure in the document's currency. Currency is
not part of the output contract, and a document mixing currencies is not detected.

### Validation: structural, semantic, and across the document

*Structural* (`models.py`, `pipeline.py`). The pydantic model forbids extra keys and coercion,
bounds confidence to [0, 1], refuses NaN, empty strings and non-ISO dates, and requires
confidence 0.0 with a null. The exact dictionary to be printed is then validated against the
published JSON Schema. The schema and the model are written separately, so
`tests/test_contract.py` feeds 27 broken documents to both and requires the same verdict.

*Semantic and business* (`fields.py`, `validate.py`). Checks come in two strengths, and the
difference is the policy:

- A **conflict** means the document, or the document and the model, give two answers for one
  field. The field becomes null. Choosing one would be a guess, and the brief asks not to pick
  a side quietly.
- A **suspicion** means the value stands and a person should look.

| Check | Code | Outcome |
|---|---|---|
| Sign against the figure, the wording or the contract sums | `direction_contradicts_figure`, `direction_contradicts_wording`, `sign_contradicts_contract_sums`, `decrease_not_shown_by_document`, `wording_gives_both_directions` | conflict |
| Amount is a contract total, a part, a ceiling, a quantity or a rate | `figure_is_a_contract_total`, `figure_is_a_part_of_the_change`, `figure_is_a_ceiling_or_an_estimate`, `figure_is_a_quantity_or_a_rate` | conflict |
| Another figure is labelled as the change; a requested figure or duration when another was granted | `another_figure_is_labelled_as_the_change`, `a_different_figure_was_approved`, `a_different_duration_was_granted` | conflict |
| Two differing values under the same kind of label: two change amounts, two dates of the document's own, two schedule durations | `two_figures_are_labelled_as_the_change`, `two_dates_are_labelled_as_the_documents_own`, `two_durations_are_given_for_the_schedule` | conflict |
| Amount against the contract sums before and after | `figure_contradicts_contract_sums` | conflict |
| Two change order numbers; a number that differs from the labelled one | `conflicting_change_order_numbers`, `number_differs_from_labelled_change_order_number` | conflict |
| Approval stated two ways (an APPROVED stamp and a rejection) | `conflicting_approval_evidence` | conflict |
| Value under another field's label | `labelled_as_something_else`, `date_belongs_to_another_event`, `duration_is_not_about_the_schedule` | conflict |
| Value not supported | `quote_not_found_in_document`, `value_not_in_quoted_text`, `wording_not_in_document`, `status_not_shown_by_quoted_text` | conflict |
| Change order dated before its contract; year outside 1990 to 2100 | `date_before_contract_date`, `date_outside_plausible_range` | suspicion |
| Qualified approval | `approval_is_qualified` | suspicion |
| The model called a field absent that the document seems to hold | `possibly_missed_value` | suspicion |
| Page with no text layer; reading order not rebuilt | `unread_pages`, `reading_order_not_rebuilt` | suspicion |
| Text addressed to a model | `instruction_like_text`, `model_reported_instructions` | suspicion |

**Cross-field rules are few on purpose.** An approved change order can be a credit. A pending
or rejected one can state an amount. A draft can have complete terms. None of these is checked,
and tests assert that they raise nothing. There is no maximum amount and no maximum duration:
the only plausibility checks compare a value with the document's own figures and dates.

## 5. Confidence: method and limits

The model is never asked how sure it is. Each accepted value starts from what the checks
established and moves down one band per caveat (`confidence.py`):

| Value | Band | Meaning |
|---|---|---|
| 0.97 | corroborated | Passed every check, and an independent one agrees: the contract sums before and after differ by exactly this amount; a duration is written in words and in figures that match |
| 0.90 | supported | Passed every check: verbatim evidence, read by rule, a label that fits, nothing competing |
| 0.75 | qualified | One caveat: approximate quote; day/month order from other dates; two-digit year; weeks converted; N/A read as zero; approval from a signature alone, or qualified; an unread page or text addressed to the model somewhere in the document |
| 0.50 | weak | Two or more caveats. Below the review threshold of 0.70 |
| 0.00 | no value | Null, whether absent or unresolved. The two are told apart internally, and only "unresolved" asks for review by itself |

The numbers are labels, chosen before any measurement and never fitted. Four values were
chosen over a continuous score because nothing here could justify more resolution than that.

### What does 0.90 mean, and what evidence says it deserves that number?

It means: this value is the document's own text, found inside a quote that code located, read
by a fixed rule, under a label that fits its field, with no competing candidate and no failed
check, and without independent confirmation. It is a statement about which checks passed. It
is not a claim that 90% of such values are correct.

The evidence, all of it on synthetic documents written for this project:

- In the recorded runs, 245 values were published at 0.90 and all 245 match their labels (33 in
  the samples, 212 in the edge cases). The 95% Wilson interval for that is 98.5% to 100%.
- The other bands are thin: 7 values at 0.97 (interval 65% to 100%), 26 at 0.75 (87% to 100%),
  none at 0.50.
- Under the stress test, the one injected mistake that was published went out at 0.90. None
  went out at 0.97.

What this does not establish:

- **Calibration.** Since every published value was correct, observed accuracy is 100% in every
  band and the bands under-claim on this data (expected calibration error about 0.11). That is
  a description of 34 synthetic documents, not a calibration result. Thirty-four documents by
  one author cannot support a probability.
- **That the bands are ordered.** No errors means no evidence that 0.97 is more reliable than
  0.90, or 0.90 than 0.75. The ordering is by construction (a higher band has passed strictly
  more checks) and is unmeasured.
- **Anything about real documents.** A value at 0.90 is exactly as reliable as the checks are
  complete. The stress test shows what they cover and one thing they do not: a name taken from
  running text, where there is no label to check.

To earn a calibrated number one would need a few hundred real change orders labelled by people
who did not build the pipeline, and then either report accuracy per band with intervals, or
fit a small model from the evidence features to correctness and report its reliability curve.

## 6. How `needs_review` is decided

`needs_review` is true when any finding marked for review exists. Confidence is one input, not
the mechanism.

1. **A field is unresolved**: a conflict, an unsupported value, an ambiguity the rules refuse,
   or the model reported `unclear`.
2. **A critical field is null.** `change_order_number`, `amount` and `approval_status`: without
   them a change order cannot be filed, priced or acted on. Also when neither a project name
   nor a contract number ties it to a contract (one of the two is enough).
3. **An absence the document seems to contradict**: the model said absent, and a scan finds a
   figure labelled as the change, a duration beside schedule wording, a labelled change order
   number, a date labelled as the document's own, or a stated status. A false "absent" is
   otherwise invisible.
4. **The document was not read whole or carries text aimed at the model**: an image-only page,
   a failed layout extraction, instruction-like text.
5. **A suspicion**: a date before the contract's, a qualified approval.
6. **An accepted value is below 0.70.**

`date`, `requested_by`, `description` and `schedule_impact_days` may be null without review
when their absence is confirmed: many change orders name no requester and state no schedule
effect.

Two consequences, both tested: a document can be valid JSON with every value at 0.90 and still
need review (a change order dated before its contract), and a value at 0.97 can sit in a
document that needs it for another field.

## 7. Evaluation

### Method

- **Labels first.** The six samples and 28 edge cases were written and labelled by hand, and
  committed, before any model read them (commits `d1866e4` and `99b7391`, both ahead of the
  pipeline's). No label was derived from an output.
- **Per field, five outcomes**: `correct_value`, `correct_null`, `wrong_value`, `invented`,
  `missed`. A wrong or invented value is the error that matters. A missed value is an
  abstention, and whether it was flagged is counted separately.
- **Outputs cannot be edited.** Each committed output has the model's raw replies beside it.
  `make eval-replay`, and the test suite, rebuild every output from those replies with the
  current code and compare byte for byte. The report is rebuilt and compared the same way.
- **One live run per document.** The samples ran on `anthropic/claude-sonnet-5.5`. The key's
  credit ran out during the first pass over the edge cases (the pipeline reported it as a
  failed run, HTTP 402, and wrote nothing for those documents), so the edge cases were run
  once on the much cheaper `anthropic/claude-haiku-5.5`. After that, each change to the
  deterministic checks was applied by rebuilding outputs from the recorded replies
  (`rebuilt_from_recorded_replies` in each diagnostics file). The prompt did not change after
  the recorded runs; a test pins its fingerprint, and replay refuses a recording made under
  another one.

### Results

| | Six samples | 28 edge cases |
|---|---|---|
| Fields correct (value or null) | 53 of 54 (98.2%) | 247 of 252 (98.0%) |
| Published values correct | 47 of 47 | 231 of 231 |
| Wrong values published | 0 | 0 |
| Values published where the document has none | 0 of 6 | 0 of 16 |
| Stated values left null | 1 of 48 | 5 of 236 |
| Documents flagged for review | 3 of 6 | 12 of 28 |
| Flagged with reason, by the labels | 3 | 8 |
| Flagged without need | 0 | 4 |
| Needed review and passed | 0 | 0 |
| Wrong values in documents passed without review | 0 of 26 | 0 of 142 |

The six abstentions, each in a flagged document:

| Document | Field | Why |
|---|---|---|
| `co-06` | `approval_status` | The approval is on a scanned page with no text layer. Reported as an unread page; no approval inferred from "signatures appear on page 2". This is the intended behaviour. |
| `e05` | `requested_by` | The model's value was not inside the line it quoted. Refused. |
| `e05` | `approval_status` | "Please countersign below to accept" is not in the pending lexicon. |
| `e23` | `amount` | The amount is written only in words. |
| `e25` | `approval_status` | "Owner acceptance: ..., by C. Dunne" is a noun label followed by a name. The lexicon does not take that for a signature. |
| `e27` | `schedule_impact_days` | The label says N/A is zero; the model reported it unclear. The label is the debatable one. |

The last four are the four unnecessary reviews. They are the cost of narrow lexicons: an
unfamiliar phrasing is refused, not guessed.

### What the checks do when the model is wrong

Neither model made a mistake that reached an output, so those runs say nothing about what a
mistake would do. `stress.py` makes them on purpose: it changes one field of each recorded
reply the way a model plausibly errs, and runs the deterministic stages. No model is called.

| Mistake played (551 in all) | Played | Refused | Published | Published with label checks off |
|---|---|---|---|---|
| Another figure taken for the amount | 71 | 71 | 0 | 24 |
| The amount's direction reversed | 29 | 29 | 0 | 0 |
| A figure that is not in the document | 29 | 29 | 0 | 0 |
| Another date taken for the change order's | 43 | 43 | 0 | 39 |
| The schedule's direction reversed | 20 | 20 | 0 | 0 |
| Another duration taken for the schedule impact | 3 | 3 | 0 | 3 |
| Another status claimed on the same quotes | 120 | 120 | 0 | 0 |
| Approval read from wording about approval | 22 | 22 | 0 | 0 |
| The contract number taken for the change order number | 32 | 32 | 0 | 9 |
| The requester taken for the project name | 33 | 32 | 1 | 32 |

A stated value reported as absent was played 149 times: 148 sent the document to review for
that field, and 1 passed unnoticed (a `0` in the days column of a pasted table row).

The one published mistake is "Owner", taken from "At the Owner's request, delete ..." as the
project name. The value is in running text with no label beside it, so there is nothing to
check it against. The document was flagged for other reasons.

### How to read this

- **These are development results.** The first stress run, against a verifier that checked
  quotes and readings but few labels, let many of these mistakes through. The label checks of
  section 3 were written in response, and the lexicons were then adjusted until the recorded
  runs and the stress test agreed. So the table records what the checks now cover; it is not
  an independent test of them, and the ablation column shows what a quote-only verifier does. The same applies to the edge cases: a wrapped duration, the forms
  of "signed", a remark after a figure and a code inside a label were all fixed after a case
  failed.
- **The documents are synthetic and share an author with the pipeline.** They cover the
  problems the brief names. They do not cover what nobody thought of.
- **The edge cases over-represent trouble**, so the 43% review rate says nothing about a real
  intake.
- **The injected mistakes are a guess** at what models get wrong. The shares are not error
  rates.
- **A live model is not deterministic.** No temperature is sent unless `LLM_TEMPERATURE` is
  set, because the provider lists none for the models used. Another run can differ; the
  recorded replies are what make the committed outputs reproducible.

## 8. Failure modes

| Failure | Handling |
|---|---|
| File missing, empty, an image, an Office file, binary | `DocumentError`, exit 3, nothing on stdout |
| PDF encrypted or damaged | `DocumentError` (`encrypted`, `malformed`) |
| Scanned PDF: no text layer | `DocumentError` (`no_text`), with the pages named. Never an empty success |
| A scanned page among text pages | Read what can be read; `unread_pages` finding; every confidence one band lower; review |
| Text layer garbled (undecodable characters, almost no letters) | `DocumentError` (`unreadable_text`) |
| Document over a limit (20 MB, 40 pages, 150,000 characters) | Refused, never cut |
| Table stored column by column in the PDF | Layout-mode extraction rebuilds rows; if it fails, plain extraction with a `reading_order_not_rebuilt` finding and review |
| Text file in Windows-1252 | Decoded, with the guess recorded |
| Model not configured | Exit 4, naming the variables, before any call |
| Key refused, no credit, model not found | `LLMUnavailableError`, no retry |
| Rate limit, 5xx, dropped connection | One retry after the wait the server asks for (20 s at most), inside the wall; then an error |
| Timeout | One wall for the whole call (default 120 s), enforced while the response streams; no retry |
| Provider without strict schemas | Stepped down to `json_object`, then no format; logged and recorded |
| Reply is not JSON or not the asked shape | One correction naming the problem; then an error |
| Reply cut off at the output limit | Error naming `LLM_MAX_OUTPUT_TOKENS`; not retried |
| A value the model made up | Its quote is not found, or the value is not in it: null, review |
| The model picks the wrong figure, date or status | Label, sign, arithmetic and cue checks: null, review (section 7 for how often) |
| The model reports a stated value absent | Absence checks: review |
| Text addressed to the model | Fenced and found: review, lower bands, and values read from it refused |
| The result would break the schema | `ContractError`, exit 5: a bug, never printed |

No error message carries the key, the request or the provider's text: a provider can echo the
document in an error body.

## 9. Simplifications made for this assignment

- **No OCR.** The brief asks for this to be considered deliberately. A text-layer reader plus a
  hard refusal for scanned files and an explicit `unread_pages` finding for scanned pages gives
  every document an honest outcome with no new dependency. An OCR fallback needs a system
  binary (Tesseract was not available on the development machine, so it could not have been
  verified), and text read by OCR would need its own, lower evidence tier. It is the first
  thing section 10 adds.
- **Lexicons, not learning.** Labels, direction words and approval cues are English regular
  expressions, written narrow. They are explainable and testable, and they are the main source
  of unnecessary reviews.
- **One record per document.** A file holding several change orders is not split.
- **No per-page deadline for PDF parsing.** A hostile PDF can make a parser loop; here the size
  and page caps are the only bound.
- **One synchronous call, no cache, no batch.** `make eval` uses four threads and nothing more.
- **A small, synthetic evaluation set**, used for development.
- **Diagnostics are a file**, not a store: no audit trail beyond what the caller keeps.
- **Not handled, and refused where they occur:** amounts in words; three-decimal currencies;
  tables with no header row; working days converted to calendar days; schedule impact given
  only as a new completion date; currency reported with the amount.

## 10. What a production deployment would change

- **Real, independently labelled documents**: a few hundred, labelled by people who do not
  maintain the pipeline, split into a development set and a held-out set that is scored rarely.
  This is the precondition for any claim about accuracy or calibration.
- **Calibrate or keep bands.** With that data, report accuracy per band with intervals and keep
  the ordinal bands, or fit a small model from the evidence features to correctness. Set the
  review threshold from the measured cost of a wrong value against the cost of a review.
- **OCR** behind the reader, with its own evidence tier and per-page confidence, and review
  forced for anything read that way until measured.
- **Read the layout, not only the text**: positions for labels and values, ticked boxes drawn as
  graphics, signatures as images. Most of what the lexicons cannot settle is visible on the page.
- **Review as a product**: show each value beside its evidence span, record the reviewer's
  correction, and feed corrections back as labelled data and as lexicon gaps.
- **Operations**: a deadline around parsing in a separate process; idempotent runs keyed by
  document hash, prompt fingerprint and model; stored diagnostics; metrics on review rate,
  reason codes and abstentions per field; alerts when the mix of reasons shifts.
- **Model management**: pin a model version, replay the recorded set before changing model or
  prompt, and run a second model only where disagreement would pay for itself.
- **Contract evolution**: currency, the evidence span and the review reasons are what consumers
  ask for next. They would go into a versioned second schema, never silently into this one.
- **Security**: the document is untrusted; keep the model without tools and without access to
  anything but the text, as it is here.

## Reuse from the Toolkit

Three cores were copied unchanged with their tests (quote matching, the decimal rule, the
injection sanitizer) and six modules follow a Toolkit design without copying it.
`src/changeorder/vendor/README.md` lists each, what was kept and what was dropped. The project
installs and runs without access to the Toolkit.

## Known unresolved issues

- Four unnecessary reviews in the edge cases, from lexicon gaps (section 7).
- A name in running text is not checked against a label, because there is none.
- A zero in a table column is not noticed when the model reports the field absent.
- The edge cases were run on a cheaper model than the samples, for budget reasons. They have
  not been run on the samples' model.
- Nothing has been measured on a real change order.
