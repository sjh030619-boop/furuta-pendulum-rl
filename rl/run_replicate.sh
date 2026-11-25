#!/usr/bin/env bash
# Independent replication of the timed stages, started at a wall-clock time.
# Two full repeats give a repeatability estimate, and the calib_ms recorded in every run
# says afterwards which repeat (if either) was measured on a busy machine.
#
#   bash rl/run_replicate.sh <epoch_seconds> [first_chain_output_file]
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
TARGET="${1:?target epoch required}"
FIRST="${2:-}"

echo "replicate armed: target $(date -d "@$TARGET" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$TARGET")"
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 60; done
echo "woke at $(date '+%F %T')"

# Never overlap with the first chain - that would contaminate both.
if [ -n "$FIRST" ] && [ -f "$FIRST" ]; then
  GIVEUP=$(( $(date +%s) + 5400 ))
  while ! grep -q "##### done" "$FIRST"; do
    if [ "$(date +%s)" -ge "$GIVEUP" ]; then echo "first chain still running after 90 min; starting anyway"; break; fi
    sleep 60
  done
fi
sleep 20

echo
echo "##### machine check before replicate"
$PY - <<'EOF'
import numpy as np, time
best = 1e9
for _ in range(3):
    a = np.random.default_rng(0).standard_normal((256, 256))
    t = time.perf_counter()
    for _ in range(140):
        a = a @ a / 64.0
    best = min(best, (time.perf_counter() - t) * 1000)
print("calib: %.0f ms  (quiet ~28, loaded ~108)" % best)
EOF

echo
echo "##### replicate stage 1: main benchmark"
$PY rl/run_all.py --budget 180 --eval-every 5 --seeds 0 1 2 3 --out rl/runs_b

echo
echo "##### replicate stage 2: UTD = 1 arm"
$PY rl/run_all.py --algos sac tqc crossq --seeds 0 1 2 3 --budget 180 --eval-every 5 \
    --kw '{"gradient_steps":8}' --tag utd1 --out rl/runs_utd1_b

echo
echo "##### figures: repeat A, repeat B, and pooled"
$PY rl/plot_results.py --runs rl/runs   --out rl/figures --name furuta_rl_wallclock_A
$PY rl/plot_results.py --runs rl/runs_b --out rl/figures --name furuta_rl_wallclock_B
$PY rl/plot_results.py --runs rl/runs rl/runs_b --out rl/figures --name furuta_rl_wallclock
$PY rl/plot_results.py --runs rl/runs rl/runs_b rl/runs_utd1 rl/runs_utd1_b \
    --out rl/figures --name furuta_rl_utd \
    --only sac sac@utd1 tqc tqc@utd1 crossq crossq@utd1 \
    --title "합성 off-policy 알고리즘: 설계 조건(UTD 1) vs 벽시계 최적(UTD 0.125)"

echo
echo "##### contamination check across both repeats"
$PY - <<'EOF'
import json, glob, os
rows = []
for d in ("rl/runs", "rl/runs_b", "rl/runs_utd1", "rl/runs_utd1_b"):
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows.append((d, r["algo"], r["seed"], r.get("calib_ms_before", 0),
                     r.get("calib_ms_after", 0), r.get("own_cpu_share", 0), r["fps"]))
if rows:
    print("%-16s %-7s %4s %9s %9s %9s %7s" % ("dir", "algo", "seed", "calib_in", "calib_out", "cpu", "fps"))
    for d, a, s, c0, c1, cpu, fps in rows:
        flag = "  <-- busy" if max(c0, c1) > 45 else ""
        print("%-16s %-7s %4d %9.0f %9.0f %9.2f %7d%s" % (d, a, s, c0, c1, cpu, fps, flag))
EOF

echo
echo "##### replicate done"
