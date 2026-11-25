#!/bin/bash
cd ~/rev11_s2r
OUT=pi_host/logs/stage5_20261124
mkdir -p $OUT
ARMS=(base effort armhome quiet bonus)
for ((r=0; r<3; r++)); do
  for ((k=0; k<5; k++)); do
    a=${ARMS[$(( (r + k) % 5 ))]}
    if [ -f STOP_STAGE2 ]; then echo "STOP 파일 -> 중단"; exit 0; fi
    python3 pi_host/wait_still.py 120 || { echo "정지 대기 실패 -> 중단"; exit 0; }
    log=$OUT/r${r}__${a}__$(date +%H%M%S).csv
    echo "=== 회차 $r  $a  $(date +%T)"
    python3 pi_host/run_policy.py --run --policy pi_host/policies_s2r/stage5/$a.npz --tau 2.52 --iq-max 120 --seconds 12 --log $log 2>&1 | grep -E "종료|한계|오류|Error|중단" | tail -3
    tc=$(python3 -c "import csv;r=list(csv.DictReader(open('$log')));print(max(int(x['temp_C']) for x in r if x['temp_C']))" 2>/dev/null)
    echo "모터 온도 최대 ${tc}°C"
    if [ -n "$tc" ] && [ "$tc" -ge 65 ]; then echo "온도 65°C 이상 -> 중단"; exit 0; fi
    sleep 2
  done
done
echo "끝"
python3 pi_host/analyze_stage2.py $OUT
