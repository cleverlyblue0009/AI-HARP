# Revision 8 — how to resume (if this session ends)

Everything below is pushed to `origin/revision-8`. Long jobs run on GitHub Actions and push their own outputs to branch
`r8-ci`, so they do not depend on any session staying alive. State at the time of writing (2026-11-16, ~11:45 UTC):

| Item | State | Where |
|---|---|---|
| Pre-registration, deviations D12–D15 | done | `ADDENDUM.md`, `DEVIATIONS.md` |
| 20 STRUCT training runs | done | `train/<run>/` |
| H4 (custom simulator) | done: not supported | `analysis/h4.json` |
| H5 (Veins, Tier B 210–219) | done: not supported | `analysis/h5.json`, `analysis/holm.json` |
| Status report | done | `REVISION8_STATUS.md` |
| Experiments C and D (D15) | done: neither explains the failure (0/5 each) | `analysis/exp_cd.json` |

## Remaining steps

**None: revision 8 is complete** (2026-11-16 ~12:45 UTC). The steps below are kept as the record of how C/D were finished.

1. Check the run: https://github.com/cleverlyblue0009/AI-HARP/actions (workflow `r8-cd`). If it failed, read the failing
   shard's log. If **gate check 2** failed (`gate.json` → `check2.pass: false`), D15 says those runners contribute nothing
   and Experiments C/D move to the laptop (`LAPTOP_RUNBOOK.md` step 7, after copying the checkpoints:
   `results/revision1/checkpoints/ref_s0_ckpt_final.pt` → `checkpoints/campaign/ref/ckpt_final.pt`,
   `ref_s{1..4}_ckpt_final.pt` → `checkpoints/revision1/ref_s{k}/ckpt_final.pt`). To rerun shards on GitHub, edit
   `ci/r8-cd.json` (seeds to rerun, bump `"request"`) and push.
2. When all 10 shards are on `r8-ci`:
   ```bash
   git fetch origin r8-ci
   git checkout origin/r8-ci -- results/revision8/veins/ci_cd
   git rm -r -q --cached results/revision8/veins/ci_cd/*/bridge_logs   # decision logs are not committed (revision-7 rule)
   python -m experiments.revision8.veins.assemble_cd                  # -> veins/expC_rows.jsonl, expD_rows.jsonl
   python -m experiments.revision8.analysis.h5 expcd                  # -> analysis/exp_cd.json
   python -m pytest tests/ -q                                         # must pass before the push
   git add results/revision8 && git commit -m "revision-8 Experiments C and D" && git push origin revision-8
   ```
3. Update `REVISION8_STATUS.md`: the C/D rows in "Answers", §7 (D15), §8 (remove item 1), with numbers read from
   `analysis/exp_cd.json` and `veins/exp{C,D}_rows_merge.json`. Then `python -m experiments.revision8.manifest`, commit, push.

Interpreter: NumPy 1.26.4 (`/root/r8env/bin/python` in the cloud session; `D:\aiharp-gpu\Scripts\python.exe` on the laptop).
Decision rule (pre-registered, ADDENDUM §8): "explains the failure" iff rural d=20 qualification rises from 0/5 to ≥ 3/5.
