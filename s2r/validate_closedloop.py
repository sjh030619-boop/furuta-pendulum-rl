"""통일 환경이 실기 폐루프를 재현하는지: 조원 A 의 실기 조합을 그대로 시뮬에서 돌려 비교합니다.

    python s2r/validate_closedloop.py

실기 조합: 원래 sac_rev11 정책(actor_rev11.npz) + 브리지 보상 (c), --tau 1.0, 1.2 A, 제동 5 W.
    tau_sim = 속도-토크 제한(u*1.0, thd)
    tau     = s*tau_sim + (Fc+- - 0.012)*tanh(thd/0.2) + (b - 1e-4)*thd,  s = (0.0232 + 2.358e-4)/5.2358e-3
    -> 1.2 A 상한 -> 제동 제한 -> 0.01 A 양자화   (run_policy.py 와 같은 순서)

시나리오: 조원 A 가 같은 펄스에서 낸 두 값 쌍(kit_params_*.json)을 씁니다.
    플랜트 = replay(MuJoCo 재생 맞춤: J, Fc+-, b),  보상 = bridge(실기 보상에 실제로 쓴 값)
    결합항은 0.87(이번 판정)과 1.0(CAD, 조원 A 원래 시뮬) 두 가지로.

같은 정의로 실기 기록도 셉니다.
    잡은 시각 = 그 뒤 2 s 넘게 |α| > 150° 가 이어지기 시작한 시각 (조원 A 의 정의)
    첫 시도 성공 = 처음 |α| > 150° 에 들어간 그 진입이 곧 잡은 것
"""
import os
import sys
import glob
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s2r_env import S2REnv, KT, IQ_MAX, BRAKE_W, BRAKE_ZONE, W_NO_LOAD, TWO_PI  # noqa: E402

MIR = os.path.join(HERE, "pi_mirror_20261124")
LOGS = os.path.join(MIR, "pi_host", "logs")
S_IN = (0.0232 + 2.358e-4) / 5.2358e-3
TH = math.radians(150.0)


class ActorNP:
    def __init__(self, path):
        w = np.load(path)
        self.W1, self.b1 = w["latent_pi.0.weight"], w["latent_pi.0.bias"]
        self.W2, self.b2 = w["latent_pi.2.weight"], w["latent_pi.2.bias"]
        self.Wm, self.bm = w["mu.weight"], w["mu.bias"]

    def __call__(self, obs):
        h = np.maximum(0.0, self.W1 @ obs + self.b1)
        h = np.maximum(0.0, self.W2 @ h + self.b2)
        return float(np.tanh((self.Wm @ h + self.bm)[0]))


class BridgeEnv(S2REnv):
    """command() 만 run_policy.py --comp 경로로 바꾼 환경."""

    def __init__(self, bridge, **kw):
        self.bridge = bridge
        super().__init__(**kw)

    def command(self, u):
        th, thd, al, ald = self._meas
        tau = u * 1.0
        if tau * thd > 0.0:
            lim = 1.0 * max(0.0, 1.0 - abs(thd) / W_NO_LOAD)
            tau = max(-lim, min(lim, tau))
        fc = self.bridge["Fc_pos"] if thd > 0 else self.bridge["Fc_neg"]
        tau = S_IN * tau + (fc - 0.012) * math.tanh(thd / 0.2) + (self.bridge["viscous"] - 1e-4) * thd
        lim = IQ_MAX * 0.01 * KT
        tau = max(-lim, min(lim, tau))
        phi = (al + TWO_PI) % TWO_PI - math.pi
        if abs(phi) > BRAKE_ZONE and tau * thd < 0.0 and thd != 0.0:
            lim = BRAKE_W / abs(thd)
            tau = max(-lim, min(lim, tau))
        iq = max(-IQ_MAX, min(IQ_MAX, int(round(tau / KT / 0.01))))
        return iq * 0.01 * KT


def catch_stats(t, alpha):
    a = np.abs((np.asarray(alpha) + math.pi) % TWO_PI - math.pi)
    up = a > TH
    dt = t[1] - t[0]
    need = int(round(2.0 / dt))
    t_catch = None
    for i in range(len(up) - need):
        if up[i] and up[i:i + need].all():
            t_catch = float(t[i])
            break
    first = np.flatnonzero(up)
    t_first = float(t[first[0]]) if first.size else None
    first_ok = t_catch is not None and t_first is not None and abs(t_catch - t_first) < 1e-9
    tail = int(round(2.0 / dt))
    return dict(t_catch=t_catch, first_ok=bool(first_ok), last2=float(np.mean(up[-tail:])))


def hardware():
    out = []
    for f in sorted(glob.glob(os.path.join(LOGS, "run_policy_2026112*_??????.csv"))):
        base = f[:-4]
        if not os.path.exists(base + "_fric.json"):
            continue
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size < 400 or np.max(np.abs(a["u"])) < 1e-6:
            continue
        s = catch_stats(a["t"], a["alpha"])
        s["run"] = os.path.basename(base)[11:]
        s["T"] = float(a["t"][-1])
        out.append(s)
    return out


def scenarios():
    out = []
    for f in sorted(glob.glob(os.path.join(LOGS, "kit_params_2026*.json"))):
        if f.endswith("latest.json"):
            continue
        k = json.load(open(f, encoding="utf-8"))
        if not k.get("valid") or "replay" not in k or "bridge" not in k:
            continue
        r, b = k["replay"], k["bridge"]
        out.append(dict(name=os.path.basename(f)[11:26], temp=k.get("temp_C"),
                        plant=dict(J=r["J"], Fp=r["Fc_pos"], Fm=r["Fc_neg"], b=r["viscous"]),
                        bridge=dict(Fc_pos=b["Fc_pos"], Fc_neg=b["Fc_neg"], viscous=b["viscous"])))
    return out


def summarize(rows, label):
    n = len(rows)
    ok = [r for r in rows if r["t_catch"] is not None]
    fa = sum(r["first_ok"] for r in rows)
    tc = np.array([r["t_catch"] for r in ok]) if ok else np.array([np.nan])
    l2 = np.array([r["last2"] for r in rows])
    print("%-26s %3d회 | 잡음 %2d/%-2d | 첫 시도 성공 %2d/%-2d (%3.0f%%) | 잡은 시각 중앙 %.2f s (%.2f~%.2f) | 마지막 2 s 직립 평균 %3.0f%%"
          % (label, n, len(ok), n, fa, n, 100.0 * fa / n, np.median(tc), np.min(tc), np.max(tc), 100 * l2.mean()))
    return dict(n=n, caught=len(ok), first_ok=int(fa), t_catch_median=float(np.median(tc)),
                t_catch_range=[float(np.min(tc)), float(np.max(tc))], last2_mean=float(l2.mean()))


def main():
    actor = ActorNP(os.path.join(MIR, "pi_host", "actor_rev11.npz"))
    res = {}
    hw = hardware()
    print("실기 (브리지 보상 실행, 기록에서 다시 셈)")
    for r in hw:
        print("  %s  잡은 %s  첫 시도 %s  마지막 2 s %3.0f%%" % (
            r["run"], "%.2f s" % r["t_catch"] if r["t_catch"] is not None else "  못 잡음",
            "성공" if r["first_ok"] else "놓침", 100 * r["last2"]))
    print()
    res["hardware"] = summarize(hw, "실기")
    scen = scenarios()
    print("시나리오 %d개 (kit_params 의 replay/bridge 쌍): %s" % (len(scen), ", ".join(s["name"] for s in scen)))
    rng = np.random.default_rng(0)
    for k, kt in ((0.87, 1.0), (1.0, 1.0), (0.87, 0.95), (1.0, 0.95)):
        rows = []
        for sc in scen:
            for rep in range(8):
                env = BridgeEnv(sc["bridge"], params=dict(sc["plant"], k=k, kt=kt, lat=int(rep % 3)),
                                episode_sec=10.0, init="hang")
                env.reset(seed=rep)
                obs = env.set_state(0.0, float(rng.uniform(-0.05, 0.05)), 0.0, 0.0)
                ts, als = [0.0], [float(env.data.qpos[1])]
                for i in range(env.max_steps):
                    obs, _r, term, trunc, info = env.step(np.array([actor(obs.astype(np.float64))]))
                    ts.append((i + 1) * env.ctrl_dt)
                    als.append(float(env.data.qpos[1]))
                    if term:
                        break
                s = catch_stats(np.array(ts), np.array(als))
                s["scenario"] = sc["name"]
                rows.append(s)
        res["sim_k%.2f_kt%.2f" % (k, kt)] = summarize(rows, "시뮬 결합 %.2f 토크 %.2f" % (k, kt))
    with open(os.path.join(HERE, "validate_closedloop.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(results=res, hardware=hw), fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
