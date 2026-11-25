"""4단계 2: 시뮬이 실기의 '서 있을 때 전류'(떨림) 순위를 재현하는지.

    python s2r/stage4_chatter.py [정책.npz ...]

공칭 조건, 매달린 상태 16개 시작(evaluate.STARTS 중 θ0=0)에서 12 s, 7~12 s 구간의
평균 |iq| [A] 와 평균 |Δu| (스텝 사이 행동 변화)를 잽니다. 실기 값(2단계, 같은 구간)과 나란히 출력.
"""
import os
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s2r_env import S2REnv, NOMINAL  # noqa: E402
from policy_np import ActorNP  # noqa: E402

KT = 2.1
REAL = {"base": 0.289, "stage1_sac_s0": 0.408, "stage1_sac_s1": 0.573, "stage1_sac_s2": 0.360, "stage1_ppo_s2": 0.196}


def hold_metrics(policy, env, starts, t0=7.0, sec=12.0):
    cur, du, err = [], [], []
    n = int(round(sec / env.ctrl_dt))
    k0 = int(round(t0 / env.ctrl_dt))
    for al0 in starts:
        o = env.set_state(0.0, al0)
        up = None
        for i in range(n):
            u = policy(o)
            o, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
            if i >= k0:
                cur.append(abs(info["tau"]) / KT)
                err.append(math.degrees(info["phi"]))
                if up is not None:
                    du.append(abs(u - up))
            up = u
            if term:
                break
    return float(np.mean(cur)), float(np.mean(du)), float(np.sqrt(np.mean(np.square(err))))


def main():
    import validate_closedloop as V
    starts = [math.radians(a) for a in (30, -30, 15, -15, 5, -5, 2, -2)]
    pols = sys.argv[1:] or [os.path.join(HERE, "policies", "stage1_%s.npz" % m)
                            for m in ("sac_s0", "sac_s1", "sac_s2", "ppo_s2")]
    res = {}
    env = S2REnv(params=dict(NOMINAL), episode_sec=12.0, init="hang")
    env.reset(seed=0)
    for p in pols:
        a = ActorNP(p)
        name = os.path.splitext(os.path.basename(p))[0]
        res[name] = hold_metrics(lambda o: a(o.astype(np.float64)), env, starts)
    base = ActorNP(os.path.join(HERE, "pi_mirror_20261124", "pi_host", "actor_rev11.npz"))
    benv = V.BridgeEnv(dict(Fc_pos=NOMINAL["Fp"], Fc_neg=NOMINAL["Fm"], viscous=NOMINAL["b"]),
                       params=dict(NOMINAL), episode_sec=12.0, init="hang")
    benv.reset(seed=0)
    res["base"] = hold_metrics(lambda o: base(o.astype(np.float64)), benv, starts)
    print("%-16s %10s %10s %10s %12s" % ("정책", "시뮬 |iq|", "시뮬 |Δu|", "시뮬 RMS°", "실기 |iq|"))
    for k, (c, d, e) in res.items():
        print("%-16s %9.3f A %10.4f %9.2f° %10s" % (k, c, d, e, ("%.3f A" % REAL[k]) if k in REAL else "-"))
    with open(os.path.join(HERE, "runs_stage4", "chatter_sim.json"), "w", encoding="utf-8") as fh:
        json.dump({k: dict(iq=v[0], du=v[1], rms_deg=v[2], real_iq=REAL.get(k)) for k, v in res.items()}, fh,
                  indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
