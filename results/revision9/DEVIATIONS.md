# Revision 9 — deviations from the addendum

Numbering continues from revision 8 (last: D15). Every entry: timestamp, what changed, why, and exactly which results
had been seen.

## D16 — H6 and S2 launched together (compute order) — 2026-11-16 18:10 UTC (logged)

ADDENDUM §7 lists the order training → H6 → S2 → C1. After the 10th training run landed (7770d5c, 16:16 UTC), the H6
Veins request and the S2 request were pushed in one commit; GitHub gave the free runners to the S2 slices first, so S2
finished (17:36) before H6 (17:59). C1 was requested after H6 finished. **Effect on results: none** — the two
computations share no inputs or outputs, nothing was chosen from either, and no H6 or S2 number existed when both were
launched. Seen at launch time: the training summary only (`results/revision9/train_summary.json`). **Seen when this
entry was written (both D16 and D17 were written at 18:10 UTC):** `s2.json` and `h6.json`.

## D17 — S2 plain slice 8 re-run on a new runner (infrastructure) — 2026-11-16 17:27 UTC (logged 18:10 UTC)

Job 133436156271 (r9-eval run 43906187897, plain slice 8) hung in `actions/checkout` for > 50 min (the other 39 slices
took 4–9 min); cancelling it was refused (403). `.github/workflows/r9-eval.yml` gained an optional `slices` field in
`ci/r9-eval.json` (default: all 20, i.e. unchanged behaviour), and plain slice 8 alone was re-run (run 43918606376,
commit 9f82e40). **What a slice computes did not change.** `sweep9 merge` rejects any task whose rows differ between
slice files. Seen at that time: the other 39 S2 slice files on `r9-ci` (not analysed), 14 of 20 H6 shards on `r9-ci`
(not analysed).
