"""동역학 기반 제어기를 PPO에 붙이는 네 가지 방식을 같은 조건에서 잽니다.

    python rl/prior_rl.py --budget 50 --seeds 0 1 2

출발 제어기는 `rl/classical.py` 가 모델에서 계산한 에너지 스윙업 + LQR 입니다(설계 0.06초,
튜닝 없음). 보상·환경·PPO 설정은 벤치마크와 완전히 같고 **제어기를 어떻게 끼우느냐만** 다릅니다.

  scratch   PPO 그대로. 비교 기준.
  residual  u = clip(u_prior + s * a).  Silver et al. 2018 / Johannink et al. 2019.
            합쳐진 정책의 기울기가 잔차에만 걸리므로 제어기가 미분 불가능해도 됩니다.
  corerl    u = (lam * u_prior + a) / (1 + lam).  Cheng et al. ICML 2019.
            제어기 쪽으로 당겨 정책 업데이트의 분산을 줄이는 대신 편향을 받습니다.
  jsrl      에피소드의 앞 h 스텝을 제어기가 끌고 가고 그 뒤부터 정책이 받습니다.
            h를 줄여 **시작 상태의 커리큘럼**을 만듭니다. Uchendu et al. ICML 2023.
            h는 시간이 아니라 **성능**으로 내립니다. 지금 h에서 제어기+정책 합성 성공률이
            기준(기본 0.8, 이 저장소의 "풀었다" 정의와 같음)을 넘을 때만 다음 단계로 갑니다.
            못 넘으면 그 h에 머무릅니다. 제어기가 끄는 h 스텝도 실제 물리 연산이므로
            벽시계 비용에 그대로 들어갑니다 — h가 클 때 표본 처리량이 크게 떨어집니다.
  bc        제어기 시연으로 BC 사전학습한 뒤 평범한 PPO.
  dapg      위에 더해, 학습 중에도 시연 항을 보조 손실로 더하고 가중치를 줄입니다.
            Rajeswaran et al. RSS 2018 의 핵심(증강된 정책경사)을 보조 손실로 옮긴 형태입니다.

BC 사전학습과 시연 수집에 걸린 시간은 모두 학습 시계에 넣습니다. 공짜가 아닙니다.
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from furuta_env import FurutaEnv
from bench import EVAL_ICS, GAMMA, NET, evaluate, calib_ms

TWO_PI = 2.0 * math.pi
BASE = os.path.join(HERE, "baseline.json")

PPO_KW = dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
              policy_kwargs=dict(net_arch=dict(pi=NET, vf=NET)))

RES_SCALE = 0.5
LAM = 3.0
N_DEMO = 60000
BC_EPOCHS = 60
BC_COEF0 = 0.1


def wrap_pi(x):
    y = (x + math.pi) % TWO_PI
    if y < 0.0:
        y += TWO_PI
    return y - math.pi


class Prior:
    """에너지 스윙업 + LQR. baseline.json 에 저장된 계수를 그대로 씁니다."""

    def __init__(self, path=BASE):
        d = json.load(open(path))
        self.K = d["K"]
        s = d["swingup"]
        self.sign, self.kE = s["sign"], s["kE"]
        self.phi_sw, self.ald_sw = s["phi_sw"], s["ald_sw"]
        self.Jp, self.mgl = d["Jp"], d["mgl"]
        self.e_up = 2.0 * self.mgl

    def u(self, th, al, thd, ald):
        phi = wrap_pi(al - math.pi)
        if abs(phi) < self.phi_sw and abs(ald) < self.ald_sw:
            u = -(self.K[0] * th + self.K[1] * phi + self.K[2] * thd + self.K[3] * ald)
        else:
            E = 0.5 * self.Jp * ald * ald + self.mgl * (1.0 - math.cos(al))
            u = self.sign * self.kE * (self.e_up - E) * math.copysign(1.0, ald * math.cos(al))
        return max(-1.0, min(1.0, u))


class PriorEnv(FurutaEnv):
    """제어기를 끼우는 방식만 다른 환경. 보상·물리·관측은 벤치마크와 같습니다."""

    def __init__(self, mode="scratch", res_scale=RES_SCALE, lam=LAM, **kw):
        self.mode = mode
        self.res_scale = res_scale
        self.lam = lam
        self.h = 0
        self.prior = Prior()
        super().__init__(**kw)

    def set_h(self, h):
        self.h = int(h)
        return self.h

    def _u_prior(self):
        return self.prior.u(float(self._qpos[0]), float(self._qpos[1]),
                            float(self._qvel[0]), float(self._qvel[1]))

    def reset(self, *, seed=None, options=None):
        """jsrl: 제어기가 앞 h 스텝을 **리셋 안에서** 끌고 간 뒤 거기서 에피소드를 시작합니다.

        밖에서 끌면 정책이 고른 행동과 실제 실행된 행동이 달라져 학습 신호가 오염됩니다.
        리셋 안에서 끝내면 에이전트가 보는 모든 전이가 자기 행동의 결과입니다. 남은 지평선이
        줄어드는 것이 곧 Uchendu et al. 이 말하는 시작 상태의 커리큘럼입니다."""
        obs, info = FurutaEnv.reset(self, seed=seed, options=options)
        if self.mode == "jsrl" and self.h > 0:
            for _ in range(min(self.h, self.max_steps - 1)):
                obs, _r, term, trunc, info = FurutaEnv.step(
                    self, np.array([self._u_prior()], dtype=np.float32))
                if term:
                    return FurutaEnv.reset(self, seed=seed, options=options)
        return obs, info

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        if self.mode == "residual":
            u = np.clip(self._u_prior() + self.res_scale * a, -1.0, 1.0)
        elif self.mode == "corerl":
            u = (self.lam * self._u_prior() + a) / (1.0 + self.lam)
        else:
            u = a
        return FurutaEnv.step(self, np.array([u], dtype=np.float32))


def make(mode, **kw):
    def _f():
        return PriorEnv(mode=mode, **kw)
    return _f


def eval_guided(model, envs, h):
    """제어기가 앞 h 스텝을 끌고 간 뒤 정책이 받는 **합성 정책**을 평가합니다.

    커리큘럼을 올릴지 말지 판단하는 신호입니다. 보고용 곡선(`evaluate`)은 제어기 없이
    정책만 재므로 이것과 다릅니다 — jsrl 에서 제어기는 학습 보조일 뿐 배포에는 없습니다."""
    k = len(envs)
    n = envs[0].max_steps
    tail = int(round(2.0 / envs[0].ctrl_dt))
    obs = np.stack([e.set_state(t0, a0, 0.0, 0.0) for e, (t0, a0) in zip(envs, EVAL_ICS)])
    rets = np.zeros(k)
    upr = np.zeros((k, n))
    alive = np.ones(k, dtype=bool)
    for i in range(n):
        if i < h:
            acts = np.array([[e._u_prior()] for e in envs], dtype=np.float32)
        else:
            acts, _ = model.predict(obs, deterministic=True)
        for j, e in enumerate(envs):
            if not alive[j]:
                continue
            o, r, term, _trunc, info = e.step(acts[j])
            obs[j] = o
            rets[j] += r
            upr[j, i] = info["upright"]
            if term:
                alive[j] = False
        if not alive.any():
            break
    f = upr[:, -tail:].mean(axis=1)
    return dict(ret=float(np.mean(rets)), success=float(np.mean(f > 0.9)),
                settled=float(np.mean(f)))


class Budget(BaseCallback):
    """예산 안에서 돌리며 주기적으로 평가합니다. jsrl 이면 커리큘럼도 여기서 진행합니다."""

    def __init__(self, budget, every, box, envs, venv=None, h0=0, stages=10,
                 gate=0.8, gate_kind="success", beta=0.9, ref=0.0, t_offset=0.0):
        super().__init__(0)
        self.budget = budget
        self.every = every
        self.box = box
        self.envs = envs
        self.venv = venv
        self.h0 = int(h0)
        self.stages = int(stages)
        self.gate = float(gate)
        self.gate_kind = gate_kind
        self.beta = float(beta)
        self.ref = float(ref)
        self.stage = 0
        self.train_time = float(t_offset)
        self.mark = None
        self.next = 0.0
        self.points = []

    def _h(self):
        if not self.h0 or self.stage >= self.stages:
            return 0
        return int(round(self.h0 * (1.0 - self.stage / float(self.stages))))

    def _curriculum(self):
        """지금 h 에서 합성 정책이 기준을 넘으면 다음 단계로. 한 번에 여러 단계도 올라갑니다.

        단계를 올릴 때마다 다시 재므로, 쉬운 상단 단계들은 학습 없이 바로 통과합니다.

        기준은 두 가지입니다. `success` 는 이 저장소의 "풀었다" 정의(성공률 ≥ 0.8)를 그대로
        쓰고, `return` 은 원 논문 쪽에 가깝게 **제어기 혼자의 리턴**의 beta 배를 넘는지 봅니다.
        전자는 5개 초기조건 중 4개라는 이산 기준이라 h 가 클 때 지나치게 가혹합니다."""
        last = None
        while self.stage < self.stages:
            h = self._h()
            g = eval_guided(self.box["model"], self.envs, h)
            last = dict(h_eval=h, g_succ=g["success"], g_ret=round(g["ret"], 2))
            if self.gate_kind == "return":
                ok = g["ret"] >= self.beta * self.ref
            else:
                ok = g["success"] >= self.gate
            if not ok:
                break
            self.stage += 1
        self.venv.env_method("set_h", self._h())
        return last

    def _on_training_start(self):
        self.mark = time.perf_counter()
        if self.h0:
            self.venv.env_method("set_h", self._h())
        self._snap()

    def _snap(self):
        self.train_time += time.perf_counter() - self.mark
        g = self._curriculum() if self.h0 else None
        m = evaluate(self.box["model"], self.envs, None)
        m["t"] = round(self.train_time, 3)
        m["steps"] = int(self.num_timesteps)
        m["h"] = self._h()
        m["stage"] = self.stage
        if g is not None:
            m.update(g)
        self.points.append(m)
        self.next = self.train_time + self.every
        self.mark = time.perf_counter()

    def _on_step(self):
        e = self.train_time + (time.perf_counter() - self.mark)
        if e >= self.budget:
            self._snap()
            return False
        if e >= self.next:
            self._snap()
        return True


class DAPG(PPO):
    """PPO 손실에 시연 항을 더하고 가중치를 줄입니다.

    원 논문은 정책경사 자체를 증강하지만, 여기서는 같은 효과를 보조 손실로 구현합니다.
    가중치는 학습이 진행되며 0으로 줄어듭니다(논문의 감쇠와 같은 취지)."""

    def set_demo(self, obs, act, coef0, horizon):
        self._d_obs = torch.as_tensor(obs, dtype=torch.float32)
        self._d_act = torch.as_tensor(act, dtype=torch.float32)
        self._coef0 = float(coef0)
        self._horizon = float(horizon)

    def train(self):
        super().train()
        if not hasattr(self, "_d_obs"):
            return
        frac = min(1.0, self.num_timesteps / self._horizon)
        coef = self._coef0 * max(0.0, 1.0 - frac)
        if coef <= 0.0:
            return
        idx = torch.randint(0, self._d_obs.shape[0], (1024,))
        o, a = self._d_obs[idx], self._d_act[idx]
        mean = self.policy.get_distribution(o).distribution.mean
        loss = coef * F.mse_loss(mean, a)
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()


def collect_demos(env_kw, n, seed):
    """제어기로 굴려 (관측, 행동) 쌍을 모읍니다."""
    env = PriorEnv(mode="scratch", **env_kw)
    obs, _ = env.reset(seed=seed)
    O, A = [], []
    while len(O) < n:
        u = env._u_prior()
        O.append(np.asarray(obs, dtype=np.float32).copy())
        A.append([u])
        obs, _r, term, trunc, _i = env.step(np.array([u], dtype=np.float32))
        if term or trunc:
            obs, _ = env.reset()
    return np.array(O, dtype=np.float32), np.array(A, dtype=np.float32)


def bc_pretrain(model, O, A, epochs, seed):
    o = torch.as_tensor(O)
    a = torch.as_tensor(A)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = o.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 1024):
            j = perm[i:i + 1024]
            mean = model.policy.get_distribution(o[j]).distribution.mean
            loss = F.mse_loss(mean, a[j])
            opt.zero_grad()
            loss.backward()
            opt.step()


MODES = ["scratch", "residual", "corerl", "corerl9", "jsrl", "bc", "dapg"]
ENVMODE = {"scratch": "scratch", "residual": "residual", "corerl": "corerl",
           "corerl9": "corerl", "jsrl": "jsrl", "bc": "scratch", "dapg": "scratch"}
LAMBDA = {"corerl": 3.0, "corerl9": 9.0}


def run_one(mode, seed, budget, every, env_kw, n_envs, vec, stages=10, gate=0.8,
            gate_kind="success", beta=0.9):
    em = ENVMODE[mode]
    ekw = dict(env_kw)
    if mode in LAMBDA:
        ekw["lam"] = LAMBDA[mode]
    fns = [make(em, **ekw) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)
    venv.seed(seed)
    evs = [PriorEnv(mode=em, **ekw) for _ in EVAL_ICS]
    for j, e in enumerate(evs):
        e.reset(seed=50_000 + j)

    cls = DAPG if mode == "dapg" else PPO
    t_pre = 0.0
    model = cls("MlpPolicy", venv, gamma=GAMMA, seed=seed, device="cpu", verbose=0, **PPO_KW)

    if mode in ("bc", "dapg"):
        t0 = time.perf_counter()
        O, A = collect_demos(env_kw, N_DEMO, seed)
        bc_pretrain(model, O, A, BC_EPOCHS, seed)
        if mode == "dapg":
            model.set_demo(O, A, BC_COEF0, 300_000)
        t_pre = time.perf_counter() - t0

    h0 = evs[0].max_steps - 1 if mode == "jsrl" else 0
    ref = json.load(open(BASE))["points"][0]["ret"] if mode == "jsrl" else 0.0
    cb = Budget(budget, every, {"model": model}, evs, venv=venv, h0=h0,
                stages=stages, gate=gate, gate_kind=gate_kind, beta=beta,
                ref=ref, t_offset=t_pre)
    model.learn(total_timesteps=10 ** 9, callback=cb)
    venv.close()

    pts = cb.points
    solved = next((p["t"] for p in pts if p["success"] >= 0.8), None)
    return dict(mode=mode, seed=seed, budget=budget, prep_s=round(t_pre, 3),
                train_time=cb.train_time, steps=int(model.num_timesteps), t_solve=solved,
                succ0=pts[0]["success"], ret0=pts[0]["ret"], h0=h0,
                stage=cb.stage, stages=cb.stages, h_end=cb._h(),
                gate_kind=cb.gate_kind, gate=cb.gate, beta=cb.beta, gate_ref=cb.ref,
                final_succ=pts[-1]["success"], final_ret=pts[-1]["ret"], points=pts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=50.0)
    p.add_argument("--eval-every", type=float, default=2.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--modes", nargs="+", default=MODES)
    p.add_argument("--armature", type=float, default=5.0e-3)
    p.add_argument("--n-envs", type=int, default=24)
    p.add_argument("--vec", default="subproc")
    p.add_argument("--stages", type=int, default=10)
    p.add_argument("--gate", type=float, default=0.8)
    p.add_argument("--gate-kind", default="success", choices=["success", "return"])
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--out", default=os.path.join(HERE, "runs_prior"))
    a = p.parse_args()

    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    env_kw = dict(arm_armature=a.armature)
    t_all = time.perf_counter()
    print("calib before: %.0f ms" % calib_ms())

    n = len(a.modes) * len(a.seeds)
    i = 0
    for mode in a.modes:
        for seed in a.seeds:
            i += 1
            c0 = calib_ms()
            print("[%2d/%2d] %-9s seed=%d ..." % (i, n, mode, seed), end=" ", flush=True)
            rec = run_one(mode, seed, a.budget, a.eval_every, env_kw, a.n_envs, a.vec,
                          stages=a.stages, gate=a.gate, gate_kind=a.gate_kind, beta=a.beta)
            rec["calib_ms_before"] = c0
            rec["calib_ms_after"] = calib_ms()
            with open(os.path.join(a.out, "%s_s%d.json" % (mode, seed)), "w") as f:
                json.dump(rec, f)
            extra = ""
            if rec["h0"]:
                extra = " stage=%d/%d h=%d" % (rec["stage"], rec["stages"], rec["h_end"])
            print("t0=%.2f/%4.0f solve=%-7s final=%.2f/%3.0f steps=%d prep=%.1fs%s"
                  % (rec["succ0"], rec["ret0"],
                     ("%.0fs" % rec["t_solve"]) if rec["t_solve"] is not None else "never",
                     rec["final_succ"], rec["final_ret"], rec["steps"], rec["prep_s"], extra))

    print("calib after: %.0f ms" % calib_ms())
    print("total %.1f min" % ((time.perf_counter() - t_all) / 60.0))


if __name__ == "__main__":
    main()
