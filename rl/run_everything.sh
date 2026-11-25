#!/usr/bin/env bash
# Full study, one stage at a time so wall-clock numbers are never contaminated by siblings.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

echo "##### stage 0: machine check"
$PY - <<'EOF'
import numpy as np, time
best = 1e9
for _ in range(3):
    a = np.random.default_rng(0).standard_normal((256, 256))
    t = time.perf_counter()
    for _ in range(140):
        a = a @ a / 64.0
    best = min(best, (time.perf_counter() - t) * 1000)
print("calib before start: %.0f ms  (quiet ~28, loaded ~108)" % best)
EOF

echo
echo "##### stage 1: main benchmark, 7 algorithms x 4 seeds x 180 s"
$PY rl/run_all.py --budget 180 --eval-every 5 --seeds 0 1 2 3 \
    --out rl/runs --save-dir rl/models

echo
echo "##### stage 2: UTD = 1 arm (composite algorithms at their design point)"
$PY rl/run_all.py --algos sac tqc crossq --seeds 0 1 2 3 --budget 180 --eval-every 5 \
    --kw '{"gradient_steps":8}' --tag utd1 --out rl/runs_utd1

echo
echo "##### stage 3: classical reference controller"
$PY rl/classical.py --out rl/baseline.json

echo
echo "##### stage 4: robustness of the trained policies"
$PY rl/robust.py --models rl/models --out rl/robust.json \
    --pick sac_s0 sac_s1 ppo_s0 ppo_s1 tqc_s0 tqc_s1 td3_s0 crossq_s0

echo
echo "##### stage 5: figures"
$PY rl/plot_results.py --runs rl/runs --out rl/figures --name furuta_rl_wallclock
$PY rl/plot_results.py --runs rl/runs rl/runs_utd1 --out rl/figures --name furuta_rl_utd \
    --only sac sac@utd1 tqc tqc@utd1 crossq crossq@utd1 \
    --title "합성 off-policy 알고리즘: 설계 조건(UTD 1) vs 벽시계 최적(UTD 0.125)"

echo
echo "##### done"
