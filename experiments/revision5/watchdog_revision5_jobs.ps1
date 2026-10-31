# Revision-5 simulation tracks for watchdog_revision5.ps1 (dot-sourced; adds to $jobs).
# Each job resumes from its CSV; "done" = the expected number of rows exists.
# Line count minus header, read in 1 MB blocks with ReadWrite sharing (works while a job appends; Get-Content took
# minutes on the 46 MB Veins JSONL and stalled every watchdog tick on 2026-10-31).
function Rows($p) {
    if (-not (Test-Path $p)) { return 0 }
    $fs = [System.IO.File]::Open((Resolve-Path $p).Path, 'Open', 'Read', 'ReadWrite')
    try { $buf = New-Object byte[] 4194304; $n = 0
          while (($r = $fs.Read($buf, 0, $buf.Length)) -gt 0) {
              $n += ([System.Text.Encoding]::ASCII.GetString($buf, 0, $r).Split([char]10)).Count - 1 } }
    finally { $fs.Close() }
    return $n - 1
}
$runs = "results\revision5\runs"

# B1. CAL classical, 24 cells: q*, m* are frozen from these (thresholds.json) before any SEL/EVAL run.
$jobs += @{ n = "cal_classical"; alive = "sweep classical --seedset CAL"; ready = { $true };
            done = { (Rows "$runs\classical__CAL__grid24__default__fallback.csv") -ge 13200 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset CAL --cells grid24 --jobs 3 >> logs\revision5\cal_classical.log 2>&1" }

# B2. Freeze q*, m* (thresholds.json) and commit it -- only that path -- before any SEL/EVAL job may start.
$thr = "results\revision5\thresholds.json"
function Frozen { if (Test-Path $thr) { git ls-files --error-unmatch $thr 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }; return $false }
$jobs += @{ n = "thresholds"; alive = "experiments.revision5.thresholds"; ready = { (Rows "$runs\classical__CAL__grid24__default__fallback.csv") -ge 13200 };
            done = { Frozen };
            cmd = "$gpu -u -m experiments.revision5.thresholds >> logs\revision5\thresholds.log 2>&1 & if exist $thr git add -f $thr && git commit -m ""revision-5: freeze q*/m* from CAL (thresholds.json), before any SEL/EVAL run"" -m ""Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"" -- $thr >> logs\revision5\thresholds.log 2>&1" }

# B3. SEL classical, then SEL rule (RAR 27 + SCF 12), 24 cells; one at a time beside training.
$jobs += @{ n = "sel_classical"; alive = "sweep classical --seedset SEL"; ready = { Frozen };
            done = { (Rows "$runs\classical__SEL__grid24__default__fallback.csv") -ge 26400 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset SEL --cells grid24 --jobs 3 >> logs\revision5\sel_classical.log 2>&1" }
$jobs += @{ n = "sel_rule"; alive = "sweep rule --seedset SEL"; ready = { (Rows "$runs\classical__SEL__grid24__default__fallback.csv") -ge 26400 };
            done = { (Rows "$runs\rule__SEL__grid24__default__fallback.csv") -ge 18720 };
            cmd = "$gpu -u -m experiments.revision5.sweep rule --seedset SEL --cells grid24 --jobs 3 >> logs\revision5\sel_rule.log 2>&1" }

# B4. SEL agent, full model seeds 0-4, 24 cells (cell-major, so the 4 headline cells finish first);
#     started early on existing checkpoints beside SEL classical (user, 2026-10-18).
$jobs += @{ n = "sel_agent_s0_4"; alive = "sweep agent --seedset SEL --cells grid24 --runs ref_s0"; ready = { Frozen };
            done = { (Rows "$runs\agent__SEL__grid24__default__fallback.csv") -ge 14400 };
            cmd = "$gpu -u -m experiments.revision5.sweep agent --seedset SEL --cells grid24 --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --jobs 2 >> logs\revision5\sel_agent_s0_4.log 2>&1" }

# B5. H3 Veins scenario bundles for the 4 headline cells, EVAL seeds 200-209 (inputs only; no replays
#     are launched until the H3 seed/H1-split question is settled -- see REVISION5 report).
$jobs += @{ n = "veins_export"; alive = "veins.queue export"; ready = { $true };
            done = { @(Get-ChildItem cache\veins\*_s2[0-9][0-9]\bundle.json -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -match '_s20[0-9]$' }).Count -ge 40 };
            cmd = "set SUMO_HOME=D:\sumo\sumo-1.19.0&& $gpu -u -m experiments.revision5.veins.queue export --seeds 200 201 202 203 204 205 206 207 208 209 >> logs\revision5\veins_export.log 2>&1" }

# B6. Custom simulator on SUMO, 4 headline cells x EVAL seeds 200-229 x 55 classical settings: the paired
#     custom side of H3's network layer (200-209) and of the 3-fold cross-fitted audit (DEVIATIONS D1).
$jobs += @{ n = "eval_classical_sumo"; alive = "sweep classical --seedset EVAL --cells headline --channel default --backend sumo"; ready = { Frozen };
            done = { (Rows "$runs\classical__EVAL__headline__default__sumo.csv") -ge 6600 };
            cmd = "set SUMO_HOME=D:\sumo\sumo-1.19.0&& $gpu -u -m experiments.revision5.sweep classical --seedset EVAL --cells headline --channel default --backend sumo --jobs 2 >> logs\revision5\eval_classical_sumo.log 2>&1" }

# B7. H3 Veins replays, network layer first: EVAL seeds 200-209, all 55 settings, 4 headline cells (2200).
#     Gated on the greedy/DV-CAST validation (flag written by hand after both validation replays pass).
# 2026-10-20: all remaining Veins replays moved to the three gated Azure VMs (experiments/revision5/cloud);
# the laptop's own queue is retired (its 1,051 clean replays are kept and on the VMs' skip list).
$jobs += @{ n = "veins_net"; alive = "veins.queue run --seeds 200 201"; ready = { $false };
            done = { $true };   # retired; all 6,600 replays merged (D3). Its old check ran Python every tick.
            cmd = "$gpu -u -m experiments.revision5.veins.queue run --seeds 200 201 202 203 204 205 206 207 208 209 --jobs 5 >> logs\revision5\veins_net.log 2>&1" }

# B8. EVAL classical and EVAL rule, 24 cells (every setting, so the bootstrap can redo the selection).
$jobs += @{ n = "eval_classical"; alive = "sweep classical --seedset EVAL --cells grid24"; ready = { (Rows "$runs\rule__SEL__grid24__default__fallback.csv") -ge 18720 };
            done = { (Rows "$runs\classical__EVAL__grid24__default__fallback.csv") -ge 39600 };
            cmd = "$gpu -u -m experiments.revision5.sweep classical --seedset EVAL --cells grid24 --jobs 3 >> logs\revision5\eval_classical.log 2>&1" }
$jobs += @{ n = "eval_rule"; alive = "sweep rule --seedset EVAL"; ready = { (Rows "$runs\classical__EVAL__grid24__default__fallback.csv") -ge 39600 };
            done = { (Rows "$runs\rule__EVAL__grid24__default__fallback.csv") -ge 28080 };
            cmd = "$gpu -u -m experiments.revision5.sweep rule --seedset EVAL --cells grid24 --jobs 3 >> logs\revision5\eval_rule.log 2>&1" }

# B9. Agent, H2 first: SEL headline seeds 5-9 -> EVAL headline seeds 5-9 -> EVAL 24 cells seeds 0-4 (cell-major,
#     so the headline cells -- H2's seeds 0-4 -- come first).
$jobs += @{ n = "sel_agent_s5_9"; alive = "sweep agent --seedset SEL --cells headline"; ready = { (Rows "$runs\agent__SEL__grid24__default__fallback.csv") -ge 14400 };
            done = { (Rows "$runs\agent__SEL__headline__default__fallback__s5_9.csv") -ge 2400 };
            cmd = "$gpu -u -m experiments.revision5.sweep agent --seedset SEL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 2 >> logs\revision5\sel_agent_s5_9.log 2>&1" }
$jobs += @{ n = "eval_agent_s5_9"; alive = "sweep agent --seedset EVAL --cells headline"; ready = { (Rows "$runs\agent__SEL__headline__default__fallback__s5_9.csv") -ge 2400 };
            done = { (Rows "$runs\agent__EVAL__headline__default__fallback__s5_9.csv") -ge 3600 };
            cmd = "$gpu -u -m experiments.revision5.sweep agent --seedset EVAL --cells headline --runs ref_s5 ref_s6 ref_s7 ref_s8 ref_s9 --tag s5_9 --jobs 4 >> logs\revision5\eval_agent_s5_9.log 2>&1" }
$jobs += @{ n = "eval_agent_s0_4"; alive = "sweep agent --seedset EVAL --cells grid24"; ready = { (Rows "$runs\agent__EVAL__headline__default__fallback__s5_9.csv") -ge 3600 };
            done = { (Rows "$runs\agent__EVAL__grid24__default__fallback.csv") -ge 21600 };
            cmd = "$gpu -u -m experiments.revision5.sweep agent --seedset EVAL --cells grid24 --runs ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 --jobs 4 >> logs\revision5\eval_agent_s0_4.log 2>&1" }

# ---- Step 1 (2026-10-31): trained-but-unevaluated runs, 4 headline cells, SEL (400-419) then EVAL (200-229),
#      biases -2..3, sampled actions. Two chains of 5 workers each. Per checkpoint: SEL 480 rows, EVAL 720 rows.
function AgentPair($name, $module, $runs, $tag, $extra, $nRuns, $after) {
    $sel = "$runs\agent__SEL__headline__default__fallback__$tag.csv"
    $ev  = "$runs\agent__EVAL__headline__default__fallback__$tag.csv"
    $nSel = 480 * $nRuns; $nEv = 720 * $nRuns
    $script:jobs += @{ n = "${name}_sel"; alive = "--seedset SEL --cells headline .*--tag $tag"; ready = $after;
        done = [scriptblock]::Create("(Rows '$sel') -ge $nSel");
        cmd = "$gpu -u -m $module agent --seedset SEL --cells headline --runs $($runs_list) $extra --tag $tag --jobs 5 >> logs\revision5\${name}_sel.log 2>&1" }
    $script:jobs += @{ n = "${name}_eval"; alive = "--seedset EVAL --cells headline .*--tag $tag"; ready = [scriptblock]::Create("(Rows '$sel') -ge $nSel");
        done = [scriptblock]::Create("(Rows '$ev') -ge $nEv");
        cmd = "$gpu -u -m $module agent --seedset EVAL --cells headline --runs $($runs_list) $extra --tag $tag --jobs 5 >> logs\revision5\${name}_eval.log 2>&1" }
    return [scriptblock]::Create("(Rows '$ev') -ge $nEv")
}
$S = "experiments.revision5.sweep"; $P = "privileged.sweep_privileged"
# chain A: ladder B -> ladder C (privileged path) -> variants
$runs_list = "rungB_s0 rungB_s1 rungB_s2 rungB_s3 rungB_s4"; $a1 = AgentPair "s1_ladderB" $S $runs "ladderB" "" 5 { $true }
$runs_list = "rungC_s0 rungC_s1 rungC_s2 rungC_s3 rungC_s4"; $a2 = AgentPair "s1_ladderC" $P $runs "ladderC" "" 5 $a1
$runs_list = "gcn_s0 gcn_s1 gcn_s2 gcn_s3 gcn_s4 mlp_s0 mlp_s1 mlp_s2 mlp_s3 mlp_s4 norel_s0 norel_s1 norel_s2 norel_s3 norel_s4"
$a3 = AgentPair "s1_variants" $S $runs "variants" "" 15 $a2
# chain B: gate tau=0 -> random gate matched to the learned gate's SEL rate -> budget checkpoints
$runs_list = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 ref_s5 ref_s6 ref_s7 ref_s8 ref_s9"
$b1 = AgentPair "s1_tau0" $S $runs "tau0" "--taus 0" 10 { $true }
$b2 = AgentPair "s1_rgate" $S $runs "rgate" "--random-gate-from SEL" 10 $b1
$runs_list = "ref_s0_u2000@1200 ref_s0_u2000@1500 ref_s0_u2000@1700 ref_s0_u2000@2000 ref_s1_u2000@1200 ref_s1_u2000@1500 ref_s1_u2000@1700 ref_s1_u2000@2000 ref_s2_u2000@1200 ref_s2_u2000@1500 ref_s2_u2000@1700 ref_s2_u2000@2000"
$b3 = AgentPair "s1_budget" $S $runs "budget" "" 12 $b2

# ---- Step 2 (2026-10-31): stress test (R9 calibrated channel, 24 cells) and SUMO (headline cells).
#      Stress thresholds are frozen and committed (that path only) before any stress SEL/EVAL job may start (D5).
$thrS = "results\revision5\thresholds_stress.json"
function FrozenS { if (Test-Path $thrS) { git ls-files --error-unmatch $thrS 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }; return $false }
$SU = "set SUMO_HOME=D:\sumo\sumo-1.19.0&&"
function Chain($name, $cmdArgs, $file, $n, $ready, $jobsN, $prefix = "") {
    $f = "$runs\$file.csv"
    $script:jobs += @{ n = $name; alive = [regex]::Escape($cmdArgs); ready = $ready;
        done = [scriptblock]::Create("(Rows '$f') -ge $n");
        cmd = "$prefix$gpu -u -m experiments.revision5.sweep $cmdArgs --jobs $jobsN >> logs\revision5\$name.log 2>&1" }
    return [scriptblock]::Create("(Rows '$f') -ge $n")
}
# classical/rule chain
$c0 = Chain "s2_st_cal" "classical --seedset CAL --cells grid24 --channel calibrated" "classical__CAL__grid24__calibrated__fallback" 13200 { $true } 2
$jobs += @{ n = "s2_st_thresholds"; alive = "thresholds --channel calibrated"; ready = $c0; done = { FrozenS };
            cmd = "$gpu -u -m experiments.revision5.thresholds --channel calibrated >> logs\revision5\s2_st_thresholds.log 2>&1 & if exist $thrS git add -f $thrS && git commit -m ""revision-5: freeze stress-test q*/m* from CAL on the calibrated channel (thresholds_stress.json, D5), before any stress SEL/EVAL run"" -m ""Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"" -- $thrS >> logs\revision5\s2_st_thresholds.log 2>&1" }
$c1 = Chain "s2_st_sel_classical" "classical --seedset SEL --cells grid24 --channel calibrated" "classical__SEL__grid24__calibrated__fallback" 26400 { FrozenS } 3
$c2 = Chain "s2_st_sel_rule" "rule --seedset SEL --cells grid24 --channel calibrated" "rule__SEL__grid24__calibrated__fallback" 18720 $c1 3
$c3 = Chain "s2_st_eval_classical" "classical --seedset EVAL --cells grid24 --channel calibrated" "classical__EVAL__grid24__calibrated__fallback" 39600 $c2 3
$c4 = Chain "s2_st_eval_rule" "rule --seedset EVAL --cells grid24 --channel calibrated" "rule__EVAL__grid24__calibrated__fallback" 28080 $c3 3
$c5 = Chain "s2_sumo_sel_classical" "classical --seedset SEL --cells headline --channel default --backend sumo" "classical__SEL__headline__default__sumo" 4400 $c4 3 $SU
$c6 = Chain "s2_sumo_sel_rule" "rule --seedset SEL --cells headline --channel default --backend sumo" "rule__SEL__headline__default__sumo" 3120 $c5 3 $SU
# agent jobs wait for the Step 1 chains (CPU)
$R10 = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4 ref_s5 ref_s6 ref_s7 ref_s8 ref_s9"; $R5 = "ref_s0 ref_s1 ref_s2 ref_s3 ref_s4"
$g1 = Chain "s2_sumo_sel_agent" "agent --seedset SEL --cells headline --channel default --backend sumo --runs $R10" "agent__SEL__headline__default__sumo" 4800 $a3 5 $SU
$g2 = Chain "s2_sumo_eval_agent" "agent --seedset EVAL --cells headline --channel default --backend sumo --runs $R10" "agent__EVAL__headline__default__sumo" 7200 $g1 5 $SU
$h1 = Chain "s2_st_sel_agent" "agent --seedset SEL --cells grid24 --channel calibrated --runs $R5" "agent__SEL__grid24__calibrated__fallback" 14400 ([scriptblock]::Create("(FrozenS) -and (& {$b3})")) 5
$h2 = Chain "s2_st_eval_agent" "agent --seedset EVAL --cells grid24 --channel calibrated --runs $R5" "agent__EVAL__grid24__calibrated__fallback" 21600 $h1 5

# ---- Step 3 (2026-10-31, D3): custom-simulator network-layer runs for H3 (2,200 paired runs, record_rx_pairs).
$jobs += @{ n = "s3_custom_h3"; alive = "veins.custom_h3"; ready = { $true };
            done = { (Rows "results\revision5\veins\custom_h3.jsonl") + 1 -ge 2200 };
            cmd = "$gpu -u -m experiments.revision5.veins.custom_h3 --jobs 2 >> logs\revision5\s3_custom_h3.log 2>&1" }
$jobs += @{ n = "s3_veins_score"; alive = "veins.veins_metrics"; ready = { $true };
            done = { (Rows "results\revision5\veins\veins_rows.jsonl") + 1 -ge 6600 };
            cmd = "$gpu -u -m experiments.revision5.veins.veins_metrics cache\veins_runs\h3 >> logs\revision5\s3_veins_score.log 2>&1" }
# rule rows on SUMO for EVAL too (only classical EVAL existed; the SUMO comparison against the best rule needs them)
$c7 = Chain "s2_sumo_eval_rule" "rule --seedset EVAL --cells headline --channel default --backend sumo" "rule__EVAL__headline__default__sumo" 4680 $c6 3 $SU

# ---- DEVIATIONS D6 (2026-10-31): sub-epoch mode calibration on audit seeds 210-219 -> freeze -> frozen-mode re-run.
$d6 = "results\revision5\veins\d6_calibration"
function D6CalDone { foreach ($cs in -95, -92, -88, -85) { foreach ($s in 4, 6, 8, 10) {
    if ((Rows "$d6\cs${cs}_sinr${s}.jsonl") + 1 -lt 200) { return $false } } }; return $true }
$jobs += @{ n = "d6_calibrate"; alive = "d6_calibrate run"; ready = { $true }; done = { D6CalDone };
            cmd = "$gpu -u -m experiments.revision5.veins.d6_calibrate run --jobs 4 >> logs\revision5\d6_calibrate.log 2>&1" }
$jobs += @{ n = "d6_score"; alive = "d6_calibrate score"; ready = { D6CalDone }; done = { Test-Path "configs\phy_patch_d6_frozen.json" };
            cmd = "$gpu -u -m experiments.revision5.veins.d6_calibrate score >> logs\revision5\d6_score.log 2>&1" }
$jobs += @{ n = "d6_rerun"; alive = "d6_calibrate rerun"; ready = { Test-Path "configs\phy_patch_d6_frozen.json" };
            done = { (Rows "results\revision5\veins\custom_h3_subepoch.jsonl") + 1 -ge 6600 };
            cmd = "$gpu -u -m experiments.revision5.veins.d6_calibrate rerun --jobs 4 >> logs\revision5\d6_rerun.log 2>&1" }
