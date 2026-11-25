# Furuta rev18b (사각 받침) — MuJoCo 간소화 모델

원본: `4_Archives/rev18b/01_CAD/Furuta_rev18b_full.step` (부품 109개). 단위는 m, kg, s.

## 구조 (강체 3개, 관절 2개)
| 바디 | 포함 부품 | 질량 | 관절 |
|---|---|---|---|
| base (고정) | 타워, 상판, 모터 고정자·모터 모듈, 슬립링, 받침·전장 | — (고정이라 동역학에 안 들어감) | 없음 |
| arm | 원기둥(drum), 뚜껑, 베어링 카트리지, 688ZZ ×2, AS5600 보드, 허브 관, 모터 출력부, 나사·인서트 | 152.6 g, 회전축 관성 Jz = 5.62e-5 kg·m² | `arm`: z축 힌지 (모터 축) |
| pendulum | SUS 축 Ø8×89, 자석 홀더·자석, 진자 허브, SUS 진자봉 Ø8×134.5, (+추) | 추 있음 111.1 g / 추 없음 100.8 g | `pendulum`: 축 방향 (cos −48°, sin −48°, 0), 피벗 z = 14 mm |

- 진자 각도 0 = 아래로 매달림, π = 도립. 키프레임 `down`, `up`.
- 질량·무게중심·관성은 STEP 솔리드 부피 × 재질 밀도로 계산(`massprops.json`, 부품별 목록 포함). 화면용 메시는 질량에 쓰지 않습니다.
- 충돌은 끔(`contype=0`). 시각용 메시는 재질별로 묶어 단순화했습니다(약 12만 삼각형).

## 파일
- `furuta_rev18b_with_tip.xml` — CAD 그대로(끝단 황동 추 Ø16×6 + 접시볼트 포함)
- `furuta_rev18b_no_tip.xml` — 추 제거(10-27 구매 결정: 추 안 달기)
- `meshes/` STL, `massprops.json`, `check_sim.py`(검증), `run_example.py`(사용 예), `tools/`(STEP → MJCF 재생성 스크립트)

## 사용
```python
import mujoco
m = mujoco.MjModel.from_xml_path('furuta_rev18b_no_tip.xml'); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('up').id)
d.ctrl[0] = 0.02            # arm 토크 [N·m], ctrlrange ±1
mujoco.mj_step(m, d)        # dt = 1 ms, RK4
arm, pend, arm_vel, pend_vel, tau = d.sensordata
```

## 검증 결과 (`check_sim.py`)
| 항목 | 추 있음 | 추 없음 |
|---|---|---|
| 질량행렬 vs 독립 라그랑주 모델(`furuta_model.py`) 최대 상대오차, 무작위 50자세 | 8.4e-6 | 5.9e-6 |
| 중력항 최대 상대오차 | 2.4e-4 | 8.4e-4 |
| 미소진동 주기 (Arm 고정) 시뮬 / 해석 | 0.6574 / 0.6572 s | 0.6141 / 0.6141 s |
| 진자 축 관성 / 피벗–COM 거리 | 5.71e-4 kg·m² / 47.8 mm | 3.60e-4 kg·m² / 38.1 mm |
| 마찰 없이 170°에서 5 s 자유운동 에너지 오차 | 1.2e-6 J | 6.2e-7 J |

적분기는 RK4입니다. implicitfast는 같은 조건에서 5 s 동안 에너지가 약 10 % 빠져서 쓰지 않았습니다.

## 실기와 맞추기 전에 정해야 하는 값 (현재는 가정값)
1. **모터 반사관성 `armature` (arm 관절) = 0** — RMD-X6-P20의 로터 관성 × 20² 가 빠져 있습니다. Arm 자체 관성(5.6e-5)이 작아서 이 값이 지배적일 가능성이 큽니다. 모터 데이터시트 값이나 실측(토크 스텝 → 각가속도)으로 넣으세요. 지금 상태로는 작은 토크에도 Arm이 매우 빨리 돕니다(`example_*.png`).
2. **Arm 쿨롱 마찰 `frictionloss` = 0.012 N·m, 감쇠 1e-4** — 10-18 시뮬에서 쓴 가정값. 감속기 마찰·백래시는 미반영.
3. **진자 마찰 2e-4 N·m, 감쇠 1e-5** — 688ZZ 2개 추정치. 자유진동 감쇠 측정으로 맞추세요.
4. **출력물 밀도 1.24 g/cm³ (인필 100 % 가정)** — 실제 인필이 낮으면 원기둥·허브 질량이 줄어듭니다.
5. 모터 출력부는 알루미늄으로 가정, 688ZZ는 카탈로그 질량 3.6 g으로 보정.
6. AS5600 12비트 양자화, 통신 지연, 토크 응답 지연은 MJCF에 없습니다. 학습 환경 래퍼에서 넣으세요.
7. 토크 한계 ±1 N·m는 학습 안전용 클램프입니다(모터 사양 아님).
8. 10-29 변경(PLA D컷 자석 홀더, 축 89 → 94 mm, 카트리지 SUS 볼트)은 이 STEP에 없습니다. 진자 쪽 질량 차이는 몇 g 수준입니다(축 +5 mm ≈ +2 g, 홀더 Al → PLA ≈ −1 g).

`tools/`로 다시 만들려면 CadQuery(OCP)와 mujoco가 필요합니다: `load_parts.py <STEP>` → `massprops.py` → `build_mjcf.py <출력폴더>`.
