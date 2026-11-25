"""기울기 시험 2: 우연 효과를 빼는 기준(0.01°)과, 세운 뒤의 질(마지막 4초)을 함께 잽니다.

  python s2r/tilt_eval2.py <정책.npz> <기울기 목록>
마지막 4초 (직립 근처 에피소드만): 진짜 직립에서 벗어난 평균 각(°), 팔 평균 속도(°/s), 팔이 한쪽으로 흘러간 정도(°/s), 평균 힘(|u|).
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilt_eval as T  # noqa: E402
import evaluate as E  # noqa: E402
from s2r_env import NOMINAL  # noqa: E402


def episode(env, pol, th0, al0):
    obs = env.set_state(th0, al0, 0.0, 0.0)
    n = env.max_steps
    phi, th, u = np.zeros(n), np.zeros(n), np.zeros(n)
    k = n
    for i in range(n):
        a = pol(obs)
        obs, r, term, _t, info = env.step(np.array([a], dtype=np.float32))
        phi[i], th[i], u[i] = info["phi"], float(env.data.qpos[0]), a
        if term:
            k = i + 1
            break
    tail = int(4.0 / env.ctrl_dt)
    if k < n or np.max(np.abs(phi[-tail:])) > math.radians(10.0):
        return None
    p = phi[-tail:]
    t = th[-tail:]
    return dict(lean=math.degrees(abs(np.mean(p))), rms=math.degrees(np.sqrt(np.mean(p ** 2))),
                drift=abs(math.degrees(t[-1] - t[0])) / 4.0, armrms=math.degrees(np.std(t)), u=float(np.mean(np.abs(u[-tail:]))))


def main():
    actor = T.ActorNP(sys.argv[1])
    pol = lambda o: actor(o.astype(np.float64))  # noqa: E731
    for tilt in [float(x) for x in sys.argv[2].split(",")]:
        for fix in (False, True):
            rows = []
            fails = 0
            for d in (0.0, 90.0, 180.0, 270.0):
                for name, over in E.CONDITIONS.items():
                    env = T.TiltEnv(tilt_deg=tilt, dir_deg=d, fix=fix, params=dict(NOMINAL, **over), episode_sec=E.EP_SEC, init="hang")
                    env.reset(seed=0)
                    for th, al in E.STARTS:
                        r = episode(env, pol, th, al)
                        if r is None:
                            fails += 1
                        else:
                            rows.append(r)
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            print("기울기 %5.2f° %-6s 마지막 4초 직립 유지 %3d/512 | 평균 기울어짐 %.2f° (RMS %.2f°) | 팔 흐름 %.1f °/s, 팔 흔들림 %.1f° | 평균 힘 %.3f" % (
                tilt, "보정" if fix else "보정없음", len(rows), m["lean"], m["rms"], m["drift"], m["armrms"], m["u"]), flush=True)


if __name__ == "__main__":
    main()
