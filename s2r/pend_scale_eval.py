"""진자를 새로 만들 때의 허용 범위: 진자 질량·길이를 바꾼 MuJoCo 에서 지금 정책이 얼마나 버티는지.

  python s2r/pend_scale_eval.py [정책.npz]

질량 배율 s_m: 진자 질량·관성 모두 s_m 배 (무게중심 거리 그대로) -> 중력 토크·결합 s_m 배, 주기 그대로.
길이 배율 s_l: 피벗-무게중심 거리 s_l 배, 관성 s_l² 배 (닮은꼴로 늘림) -> 중력 토크 s_l 배, 주기 sqrt(s_l) 배.
판정은 s2r/evaluate.py 의 엄격 기준(조건 8 x 시작 16 = 128칸).
"""
import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s2r_env import S2REnv, NOMINAL  # noqa: E402
import evaluate as E  # noqa: E402
from validate_closedloop import ActorNP  # noqa: E402

SCALE = {"m": 1.0, "l": 1.0}


class ScaledEnv(S2REnv):
    def __init__(self, **kw):
        super().__init__(**kw)
        sm, sl = SCALE["m"], SCALE["l"]
        m, p = self.model, self._pbody
        m.body_mass[p] *= sm
        m.body_ipos[p][1:] *= sl
        self._I_cad = self._I_cad * sm * sl * sl
        self._dg = self._coupling_slope()
        self._apply(self.p)


def period(env):
    import mujoco
    d = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, d)
    M = np.zeros((2, 2))
    mujoco.mj_fullM(env.model, d, M)
    mgl = env.model.body_mass[env._pbody] * 9.81 * float(np.linalg.norm(env.model.body_ipos[env._pbody][1:]))
    return 2 * math.pi * math.sqrt(M[1, 1] / mgl), mgl


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs_app_rules", "sac_s1e1.npz")
    actor = ActorNP(path)
    pol = lambda o: actor(o.astype(np.float64))  # noqa: E731
    grid = [(1.0, 1.0), (0.8, 1.0), (0.9, 1.0), (1.1, 1.0), (1.2, 1.0), (1.35, 1.0),
            (1.0, 0.9), (1.0, 1.1), (1.0, 1.2), (1.2, 1.2), (0.8, 1.2)]
    print("정책", os.path.basename(path))
    for sm, sl in grid:
        SCALE["m"], SCALE["l"] = sm, sl
        T, mgl = period(ScaledEnv(params=dict(NOMINAL), episode_sec=E.EP_SEC, init="hang"))
        r = E.evaluate(pol, env_cls=ScaledEnv, verbose=False)
        hold = [r[k]["t_hold_med"] for k in E.CONDITIONS if r[k]["t_hold_med"] is not None]
        print("  질량 x%.2f 길이 x%.2f | 주기 %.3f s, mgl %.4f N·m | 엄격 %3d/128  느슨 %3d/128  직립 시작 중앙 %s" % (
            sm, sl, T, mgl, r["total_strict"], r["total_loose"], ("%.2f s" % float(np.median(hold))) if hold else "-"), flush=True)


if __name__ == "__main__":
    main()
