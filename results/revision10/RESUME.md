# Revision 10 — resume point

**Revision 10 is complete** (2026-11-19): H7, S3, S4, S5 committed; `REVISION10_STATUS.md` final; reproduction checked.
Nothing remains to run.

If the session ends, continue from here. Branch `revision-10`; runner outputs on `r10-ci`. Interpreter with NumPy 1.26.4
(cloud: `/root/r8env/bin/python`; laptop: `AIHARP_PY`).

## Done (committed on `revision-10`)

| Step | Commit |
|---|---|
| 0–1: reading notes, seed inventory, seeds.json (R-A 230–239, R-B 240–249) | `6702c83` |
| 1: gates.json (pipeline, checkpoint, 40 bundles, Veins source check) — all pass | `10c0966` |
| 2: ADDENDUM.md pushed (before any replay on 230–249) | `27aea29`; header records it in `738dd3e` |
| 3: analysis code `experiments/revision10/analysis/veins10.py` + test, `ci10.py`, `r10-veins.yml`, reproduce scripts | `f9e4388` |
| 4a: H7 requested (`ci/r10-veins.json` mode h7, seeds 230–239) | `48f7883`, run 44368078403 |
| 4a: **H7 done**: 10/10 shards accepted; h7.json (NOT supported, 0/5 vs 0/5, p = 1.0) and s4.json | `e887d04` |
| 4b: S3 requested (mode s3, seeds 240–249) | `c72a3e4` |
| 4b: **S3 done**: 10/10 accepted; s3.json | `0839c72` |
| 4c: S5 requested (mode s5, seeds 230–239), run 44409896065 | `cdfddd5` |

## Remaining, in order (ADDENDUM §7: H7 → S3 → S5)

1. *(done)* When H7's 10 shards are on `r10-ci` (`git ls-tree -r --name-only origin/r10-ci results/revision10/veins/h7/ | grep -c
   gate.json` = 10) and the run has finished:
   ```
   git fetch origin r10-ci
   git checkout origin/r10-ci -- results/revision10/veins/h7
   git rm -r -q --cached --ignore-unmatch results/revision10/veins/h7/*/bridge_logs results/revision10/veins/h7/*/gate2_logs
   python -m experiments.revision10.analysis.veins10 assemble h7     # must accept 10/10 shards
   python -m experiments.revision10.analysis.veins10 h7
   python -m experiments.revision10.analysis.veins10 s4
   git add results/revision10/veins/h7 results/revision10/veins/h7_rows*.json* results/revision10/analysis
   git commit -m "revision-10: H7 ..."; git push origin revision-10
   ```
   A rejected shard: re-run it once on a fresh runner (bump "request", seeds = that seed only) and log a deviation
   (D18…); a gate failure on every runner or a second failure = STOP and report.
2. *(done)* S3: `ci/r10-veins.json` = `{"mode": "s3", "seeds": [240, …, 249], "request": 2}`, push; then as step 1 with `s3`, and
   `python -m experiments.revision10.analysis.veins10 s3`.
3. *(done)* S5: `{"mode": "s5", "seeds": [230, …, 239], "request": 3}`; then as step 1 with `s5`, and `veins10 s5`.
4. *(done)* `results/revision10/REVISION10_STATUS.md` (addendum hash, gates incl. the Veins analogue-model result, H7, S3, S4, S5,
   deviations, reproduction, open questions); `./reproduce_revision10.sh` must exit 0 with outputs identical apart from
   provenance; full test suite before every push. Never commit `bridge_logs/` or `gate2_logs/`.
