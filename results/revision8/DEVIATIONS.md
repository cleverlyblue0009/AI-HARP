# Revision 8 — deviations from the addendum

Numbering continues from revision 6 (last: D11). Every entry: timestamp, what changed, why, and which results had
been seen. None yet.

## D12 — Veins scope cut to fit one day (written 2026-11-16T05:31Z, decided by the author)

**Pre-registered (ADDENDUM §6, §8):** H5 Tier B on seeds 210–229 (2,400 replays), then H5 Tier A on 200–209 (1,200),
then Experiments C and D in rural d=20 and urban NLOS d=20 (600 replays each): about 55 laptop wall-hours.

**Changed to (author: "highest accuracy with minimal time"; option A of the assistant's three):**
1. **H5 Tier B on seeds 210–219** (10 seeds; 1,200 replays) instead of 210–229. Everything else in H5 is unchanged:
   both cells, both arms, training seeds 0–4, δ ∈ {−2..3}, thresholds frozen from Tier A
   (`results/revision8/veins_thresholds.json`), Algorithm 1, the per-cell rule, the Fisher test, p_H5 and Holm.
2. **H5 Tier A is not run** (it was descriptive only).
3. **Experiments C and D in rural d=20 only** (300 replays each), the only cell their pre-registered decision rule
   reads. Urban NLOS d=20 is not run for them.
4. Training order (not pre-registered, recorded here for completeness): the ten runs H5 needs (plain and rand s0–4)
   first, then plain s5–9 (H4), then rand s5–9 (descriptive).

**Why:** cost and time only; the full plan needs about 55 laptop wall-hours. The new plan needs about 21: gate about 1.5 h,
Experiments C and D about 6.5 h (they need no STRUCT training and start at once), H5 Tier B about 13 h once the ten
runs are trained. Assistant's recommendation over the alternatives: halving Tier B keeps both H5 cells and the
registered rule; the Fisher test's power depends on the 5 training seeds per arm, not on the replay seeds, but each
training seed's matched cost now rests on 10 replay seeds instead of 20 and is noisier. Renting cloud VMs for Veins (option C) was not chosen.

**Seen when this was decided:** no STRUCT evaluation of any kind and no revision-8 Veins replay exist. Seen: training
history of plain_s0 up to generation 14 (training-pool fitness only), the Exp C geometry validation (1,789 contexts,
0 mismatches), and everything listed in ADDENDUM §0.
