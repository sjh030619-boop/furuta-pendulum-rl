#!/bin/bash
cd ~/rev11_s2r
OUT=pi_host/logs/stage2_20261124
for r in 1 2 3; do
  python3 pi_host/wait_still.py 120 || { echo "정지 대기 실패 -> 중단"; exit 0; }
  log=$OUT/r${r}b__base__$(date +%H%M%S).csv
  echo "=== 재시험 $r  base  $(date +%T)"
  python3 pi_host/run_policy.py --run --policy pi_host/actor_rev11.npz --tau 1.0 --comp --iq-max 120 --seconds 12 --log $log 2>&1 | grep -vE "^\s*$" | tail -4
  sleep 2
done
echo "끝"
