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

## D13 — AI-HARP checkpoints lost: Experiments C and D and gate check 2 not run (written 2026-11-16T08:03Z, author decision)

**Pre-registered (ADDENDUM §8):** Experiments C and D replay AI-HARP ref_s0–ref_s4; the Veins gate's check 2 requires the
AI-HARP ref_s0 replay with every new option off to be byte-identical to the revision-6 replay.

**What happened:** the laptop's working copy was deleted; `checkpoints/` is git-ignored, and none of the five
checkpoints (`checkpoints/campaign/ref/ckpt_final.pt`, `checkpoints/revision1/ref_s1..ref_s4/ckpt_final.pt`,
hashes in `results/revision5/checkpoint_shas.json`) exists on the laptop, on D:, E:, in the WSL home or in the Recycle Bin
(laptop session report and the author, 2026-11-16). Retraining them bit-identically is not possible within the time.

**Changed to:** Experiments C and D are not run (they were descriptive and outside the Holm family; H4 and H5 are not
affected). Gate check 2 is dropped: it guarded only the AI-HARP path of the app, which only C and D use. The gate keeps
check 1 (classical replay byte-identical to the revision-6 gate log `707d6533…`), and the STRUCT checks (STRUCT replay
runs cleanly, every logged request reproduced offline). The question whether the beacon table or the coupling lag
explains AI-HARP's Veins failure stays open and is reported as such.

**Seen:** no STRUCT evaluation and no revision-8 Veins replay exist. Training-pool fitness histories of plain_s0
(finished) and plain_s1 (in progress) only.

## D14 — compute moved to GitHub Actions runners (written 2026-11-16T08:03Z, author decision)

**Pre-registered:** training in this cloud session; custom evaluation here; Veins replays on the laptop (ADDENDUM §4–6,
LAPTOP_RUNBOOK.md).

**Changed to (author: "very fast free option"):** the repository is public, so GitHub Actions runners (Ubuntu, 4 vCPU)
are free. (1) **Training:** the remaining runs each train on their own runner, unchanged code, budget, seeds and
configuration; plain_s0 and plain_s1 train here. Each runner records its CPU model, Python, NumPy and cma versions in
`run_meta.json`. A run is deterministic on its runner; across CPU types last-bit floating-point differences can occur
(measured between this machine and the laptop: `rwcr` ≤ 1 ulp), so a rerun on another CPU type may not be bit-identical;
this is reported. (2) **Custom evaluation** (sweep_struct) is split into shards by task hash, each row identical in
definition; the shards are merged. (3) **Veins replays (H5 Tier B, D12 scope)** run on runners with OMNeT++ 6.0.3,
Veins 5.3 and SUMO 1.19.0 built from source by the revision-5 cloud installer (`experiments/revision5/veins/install_wsl.sh`,
as on the revision-5 Azure VMs). **Every runner first runs the gate:** the classical slotted_1p n=8 replay (rural
d=20, seed 200) must be byte-identical to `707d6533…`; a runner that fails it contributes no replay. STRUCT trace checks
(offline re-send of every logged request) run on every runner's own logs. Acceptance per replay unchanged (exit 0,
identical SUMO trace, gated binary of that runner).

**Seen:** as for D13.

## D15 — D13 reversed: the AI-HARP checkpoints were never lost; Experiments C and D run on runners (written 2026-11-16T11:39Z, author decision)

**What was wrong in D13:** the five checkpoints are committed in the repository as
`results/revision1/checkpoints/ref_s{0..4}_ckpt_final.pt`; their `checkpoint_sha` values (b589be512827fcbb, 4867ffabbdf23aac,
0b5fa61c06396e28, 2c2434ab9ed05da2, 3cb2a44fef5bde6b) equal `results/revision5/checkpoint_shas.json` exactly. D13 was
written after a search of the laptop's git-ignored `checkpoints/` folder only; the assistant did not search the
repository first.

**Changed to:** Experiments C and D and gate check 2 are run as registered under D12 (rural d=20 only, Tier A seeds
200–209, AI-HARP ref_s0–ref_s4, biases −2..3, τ = 0.5), on GitHub runners as under D14: the committed checkpoint files
are copied to the paths the code reads (`experiments/revision5/sweep.py` `checkpoint_for`), and each file's hash is
checked against `checkpoint_shas.json` before any replay. Every runner first runs gate check 1 (classical replay
byte-identical to `707d6533…`) and gate check 2 (AI-HARP ref_s0, bias +1, every new option off, byte-identical to the
revision-6/7 replay `220dd1fd…`, committed in `results/revision7/veins/raw/r7_gate/`). If check 2 fails on a runner
(for example through last-bit differences of torch on Linux vs the Windows decision server that produced the reference),
that runner contributes no replay and the experiments move to the laptop as in LAPTOP_RUNBOOK.md. Decision rule unchanged
("explains the failure" iff rural d=20 qualification rises from 0/5 to ≥ 3/5).

**Seen when this was decided:** every revision-8 result: H4 (not supported), H5 (not supported), the STRUCT-rand
descriptive results and the post-hoc observation in REVISION8_STATUS.md §6. Experiments C and D concern only the existing
AI-HARP checkpoints and are descriptive and outside the Holm family; none of their replays exists.
