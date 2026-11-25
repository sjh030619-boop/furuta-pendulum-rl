"""4단계 1: 1단계 SAC 체크포인트(학습 1분마다)를 전부 128칸 평가해서 멈춤 규칙을 고릅니다.

    python s2r/stage4_ckpt.py --seed 0 [--algo ppo]   # 시드 하나 (프로세스 셋으로 나눠 돌림)
    python s2r/stage4_ckpt.py --report          # 멈춤 규칙 비교 표

학습 중 빠른 평가(공칭 16칸, runs_stage1/sac_s*.json 의 points)로 멈춘다고 할 때,
그 시점 체크포인트가 128칸에서 얼마나 되는지를 봅니다. 액터는 numpy 로 돌려 빠르게(SB3 와 출력 차이 < 1e-5).
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs_stage1")
OUT = os.path.join(HERE, "runs_stage4")


def np_actor(zip_path):
    from stable_baselines3 import SAC
    m = SAC.load(zip_path, device="cpu")
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in m.policy.state_dict().items()}
    W1, b1 = sd["actor.latent_pi.0.weight"], sd["actor.latent_pi.0.bias"]
    W2, b2 = sd["actor.latent_pi.2.weight"], sd["actor.latent_pi.2.bias"]
    Wm, bm = sd["actor.mu.weight"], sd["actor.mu.bias"]

    def pol(o):
        h = np.maximum(0.0, W1 @ o.astype(np.float64) + b1)
        h = np.maximum(0.0, W2 @ h + b2)
        return float(np.tanh((Wm @ h + bm)[0]))
    return pol


def np_actor_ppo(zip_path):
    from stable_baselines3 import PPO
    from base_train import np_policy
    return np_policy(PPO.load(zip_path, device="cpu"), "ppo")


def run_seed(seed, algo="sac"):
    import torch
    torch.set_num_threads(1)
    from evaluate import evaluate
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, "ck_%s_s%d" % (algo, seed), "t*.zip"))):
        t = int(os.path.basename(p)[1:6])
        r = evaluate(np_actor(p) if algo == "sac" else np_actor_ppo(p), verbose=False)
        rows.append(dict(t=t, strict=r["total_strict"], loose=r["total_loose"],
                         per={k: v["strict"] for k, v in r.items() if isinstance(v, dict)}))
        print("%s s%d t=%5d  128칸 엄격 %3d  느슨 %3d" % (algo, seed, t, r["total_strict"], r["total_loose"]), flush=True)
    with open(os.path.join(OUT, "ckpt_%s_s%d.json" % (algo, seed)), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)


RULES = {
    "R1 공칭 처음 ≥12/16": lambda q, i: q[i] >= 12,
    "R2 공칭 2번 연속 ≥12/16": lambda q, i: i >= 1 and q[i] >= 12 and q[i - 1] >= 12,
    "R3 공칭 처음 ≥15/16": lambda q, i: q[i] >= 15,
    "R4 공칭 2번 연속 ≥15/16": lambda q, i: i >= 1 and q[i] >= 15 and q[i - 1] >= 15,
    "R5 공칭 16/16": lambda q, i: q[i] >= 16,
}


def report():
    res = {}
    for seed in (0, 1, 2):
        ck = {r["t"]: r for r in json.load(open(os.path.join(OUT, "ckpt_sac_s%d.json" % seed), encoding="utf-8"))}
        pts = json.load(open(os.path.join(RUNS, "sac_s%d.json" % seed), encoding="utf-8"))["points"]
        q = [p["strict"] for p in pts]
        ts = [int(p["t"]) for p in pts]
        res[seed] = dict(ck=ck, q=q, ts=ts)
        full = [ck[k]["strict"] for k in sorted(ck)]
        print("s%d 128칸 곡선: %s" % (seed, " ".join(str(v) for v in full)))
    print()
    out = {}
    for name, rule in RULES.items():
        line = []
        for seed in (0, 1, 2):
            d = res[seed]
            i = next((i for i in range(len(d["q"])) if rule(d["q"], i)), None)
            if i is None:
                line.append(None)
                continue
            t = d["ts"][i]
            k = min(d["ck"], key=lambda x: abs(x - t))
            line.append((t, d["ck"][k]["strict"]))
        out[name] = line
        print("%-22s %s" % (name, "   ".join("s%d: %s" % (s, ("%d s -> %d/128" % v) if v else "없음")
                                             for s, v in zip((0, 1, 2), line))))
    with open(os.path.join(OUT, "stop_rules.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run_seed(a.seed, a.algo)


if __name__ == "__main__":
    main()
