"""진자가 매달려 멈출 때까지 기다림 (AS5600 I2C 읽기만, 모터 명령 없음).

run_policy.wait_pendulum_still 과 같은 기준: 0.5 s 동안 RAW (최대-최소) <= 6 카운트,
그리고 아래로 매달림: RAW 가 2034 (매달림 영점) ±60 카운트(±5°) 안.
(2026-11-24 추가: PPO 뒤 진자가 직립 근처에서 마찰로 멈춘 채 남아 다음 회가 거꾸로 시작한 일 대책)
    python3 wait_still.py [최대초=120]     멈추면 0, 시간 넘으면 1 로 끝남
"""
import sys
import time
from smbus2 import SMBus

lim = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
bus = SMBus(1)
t0 = time.time()
while time.time() - t0 < lim:
    v = []
    t1 = time.time()
    while time.time() - t1 < 0.5:
        hi, lo = bus.read_i2c_block_data(0x36, 0x0C, 2)
        v.append(((hi & 0x0F) << 8) | lo)
        time.sleep(0.005)
    if max(v) - min(v) <= 6 and abs(v[-1] - 2034) <= 60:
        print("진자 정지 %.1f s (RAW %d)" % (time.time() - t0, v[-1]))
        sys.exit(0)
print("진자 %.0f s 안에 아래로 멈추지 않음 (RAW %d)" % (lim, v[-1]))
sys.exit(1)
