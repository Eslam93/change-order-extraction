# Evaluation

| Path | Holds |
|---|---|
| `inputs/` | 28 short text documents, each written to probe one behaviour |
| `truth/` | A hand-written label file for each, with the reason for the label |
| `outputs/`, `diagnostics/` | What the pipeline delivered, and the recorded run behind it |
| `REPORT.md` | The scores for these and for the six samples, and the stress test. Generated |
| `results.json` | The same numbers as data |

The labels were committed before any model read the documents. The outputs are never edited.

```bash
make eval-replay    # no network: rebuild every output from its recorded replies and compare
make eval           # call the live model for every input (needs LLM_* settings), then report
uv run python -m changeorder.evaluation rebuild eval   # after changing a check: derive the
                                                        # outputs again from the recorded replies
uv run python -m changeorder.evaluation report         # rescore and rewrite REPORT.md
```

What each case probes:

| Case | Probes |
|---|---|
| `e01` | The plain case, in another layout |
| `e02` | An explicitly zero-cost change, with a contract sum on the same line |
| `e03` | No schedule mentioned: null without review; approval by a signature |
| `e04` | A credit in parentheses, and a completion date that moves earlier |
| `e05` | A credit shown only by the word; an explicit no-change schedule; an unsigned acceptance line |
| `e06` | Only a revised contract total: no amount to take |
| `e07` | A requested amount and a different approved one |
| `e08` | Nine figures in a column: lines, subtotal, markups, tax, total |
| `e09` | A date that reads two ways, with nothing to settle it |
| `e10` | A date that reads two ways, settled by the document's other dates |
| `e11` | European number format |
| `e12` | Time and material with a not-to-exceed ceiling; schedule put off |
| `e13` | Two change order numbers |
| `e14` | An approval section that was never signed |
| `e15` | A rejected request that still states its amount |
| `e16` | A draft with complete terms |
| `e17` | A voided change order that names its replacement |
| `e18` | An APPROVED stamp and a rejection in one document |
| `e19` | A paragraph addressed to the model |
| `e20` | A memo with almost nothing in it |
| `e21` | Nine dates, one of them the change order's |
| `e22` | A schedule in weeks; everything on one header line |
| `e23` | An amount written only in words |
| `e24` | Five large figures around the change |
| `e25` | Cost up, time down |
| `e26` | A row pasted from a spreadsheet, with its labels in the header row above |
| `e27` | A two-digit year; "N/A" for the schedule |
| `e28` | Two dates, neither the change order's |
