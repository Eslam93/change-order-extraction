# The six sample documents

Synthetic change orders written for this project: invented projects, companies, people and
figures. They were written to carry the problems real change orders have, and each was labelled
by hand before any model read it.

| Folder | Holds |
|---|---|
| `inputs/` | The documents: four PDFs and two pasted-text files |
| `truth/` | One label file per document: the nine values a person would record, alternatives that are also right, whether review is warranted, and why |
| `outputs/` | What the pipeline delivered for each: the contract JSON, as generated |
| `diagnostics/` | The run behind each output: the model's raw replies, the evidence found for every field, the findings, the model and prompt used |
| `build_samples.py` | Rebuilds the four PDFs. A rebuild changes no byte unless the content changed |

| Document | What it is | What it tests | Delivered |
|---|---|---|---|
| `co-01-standard-form.pdf` | A clean owner/architect form with a contract-sum summary | The plain case. Five figures, one of which is the change; a contract date, three signature dates and a completion date that are not the change order's date; a "not valid until signed" notice above a real approval | All nine fields; amount and schedule corroborated; no review |
| `co-02-contractor-request.txt` | An email from a subcontractor, pasted as text | No contract number; an itemised price with a subtotal and markup; "No change to the contract completion date" (0, not null); "valid for 15 days" (not the schedule); pending | Eight fields and a null contract number; no review |
| `co-03-deductive-credit.pdf` | A credit, with the header's two blocks written to the page one after the other | A figure in parentheses labelled DEDUCT; a subtotal of credits that is not the net change; a schedule that gets shorter; approval shown by signatures | All nine fields, amount and schedule negative; no review |
| `co-04-subcontractor-multi-amount.pdf` | A priced proposal the general contractor approved for less, with the line-item table stored column by column | A requested total and a different approved amount; tax, bond and markup lines; days requested and days granted; a date written `03/04/2026`; "Approved as noted" | The approved amount and granted days; date null; review |
| `co-05-draft-conflicting.txt` | A draft with unfinished terms | Two different change order numbers; a price "TBD" beside a not-to-exceed ceiling; contract time "to be determined"; an approval section with three empty signature lines | Number, requester, amount and schedule null; status `draft`; review |
| `co-06-scanned-signature-page.pdf` | Page 1 has a text layer; page 2, with the signatures and an APPROVED stamp, is a scan | A page the reader cannot see. The label records what a person reading both pages would: approved | Approval null; the page reported as unread; every confidence one band lower; review |

The outputs are never edited. `make eval-replay`, and the test suite, rebuild each one from the
recorded replies with the current code and compare byte for byte.

The six ran on `anthropic/claude-sonnet-5.5` through OpenRouter, in strict JSON-schema mode,
under prompt `change_order_facts@1`. Each diagnostics file records the settings of its run.
`eval/REPORT.md` scores them against the labels.
