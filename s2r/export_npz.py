"""SB3 SAC/PPO 모델(.zip) -> Pi 호스트가 읽는 .npz, 그리고 SB3 와 같은 출력을 내는지 검증.

    python s2r/export_npz.py <model.zip> <out.npz>

검증: 관측 공간에서 무작위 관측 5000개 + 통일 환경 롤아웃 관측 2000개로 SB3 predict(결정적)와
numpy 액터(s2r/policy_np.py) 출력 차이를 잼. 1e-5 를 넘으면 실패로 끝냄.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def export(src, dst):
    from stable_baselines3 import SAC, PPO
    try:
        m = SAC.load(src, device="cpu")
        kind = "sac"
    except Exception:
        m = PPO.load(src, device="cpu")
        kind = "ppo"
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in m.policy.state_dict().items()}
    if kind == "sac":
        out = {k: sd["actor." + k] for k in ("latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight",
                                             "latent_pi.2.bias", "mu.weight", "mu.bias")}
    else:
        act = type(m.policy.mlp_extractor.policy_net[1]).__name__
        if act != "Tanh":
            raise SystemExit("PPO 은닉층 활성화가 Tanh 가 아님: %s" % act)
        out = {"kind": np.array("ppo"),
               "pi.0.weight": sd["mlp_extractor.policy_net.0.weight"], "pi.0.bias": sd["mlp_extractor.policy_net.0.bias"],
               "pi.2.weight": sd["mlp_extractor.policy_net.2.weight"], "pi.2.bias": sd["mlp_extractor.policy_net.2.bias"],
               "action.weight": sd["action_net.weight"], "action.bias": sd["action_net.bias"]}
    np.savez(dst, **out)
    return m, kind


def verify(m, dst):
    from policy_np import ActorNP
    from s2r_env import S2REnv
    a = ActorNP(dst)
    rng = np.random.default_rng(0)
    th, al = rng.uniform(-np.pi, np.pi, 5000), rng.uniform(-np.pi, np.pi, 5000)
    O = np.stack([np.cos(th), np.sin(th), np.cos(al), np.sin(al),
                  rng.uniform(-1.2, 1.2, 5000), rng.uniform(-0.8, 0.8, 5000)], 1).astype(np.float32)
    env = S2REnv(randomize=True)
    o, _ = env.reset(seed=1)
    roll = []
    for _ in range(2000):
        roll.append(o)
        o, _r, te, tr, _i = env.step(m.predict(o, deterministic=True)[0])
        if te or tr:
            o, _ = env.reset()
    O = np.concatenate([O, np.array(roll, dtype=np.float32)])
    ref = m.predict(O, deterministic=True)[0][:, 0]
    mine = np.array([a(o.astype(np.float64)) for o in O])
    err = float(np.max(np.abs(ref - mine)))
    return a.kind, err, len(O)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m, kind = export(src, dst)
    k, err, n = verify(m, dst)
    print("%s -> %s  (%s) | numpy 와 SB3 출력 최대 차이 %.2e (관측 %d개)" % (src, dst, k, err, n))
    if err > 1e-5:
        raise SystemExit("검증 실패")


if __name__ == "__main__":
    main()
