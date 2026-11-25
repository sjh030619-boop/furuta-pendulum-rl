"""SB3 액터를 numpy 만으로 실행 (Pi 호스트 pi_host/policy_np.py 를 대체하는 확장판).

SAC (기존 그대로, 조원 A 형식): u = tanh(mu(ReLU(W2 ReLU(W1 obs + b1) + b2)))
    npz 키: latent_pi.0.weight/bias, latent_pi.2.weight/bias, mu.weight/bias
PPO (추가, 2026-11-24): u = clip(action(tanh(W2 tanh(W1 obs + b1) + b2)), -1, 1)
    npz 키: kind = "ppo", pi.0.weight/bias, pi.2.weight/bias, action.weight/bias
    SB3 PPO MlpPolicy 기본(은닉층 tanh, 결정적 행동 = 평균, 행동 공간으로 클리핑)과 같다.
kind 키가 없으면 SAC 로 읽으므로 기존 npz(actor_rev11.npz 등)는 그대로 동작한다.
"""
import numpy as np


class ActorNP:
    def __init__(self, path):
        w = np.load(path)
        self.kind = str(w["kind"]) if "kind" in w.files else "sac"
        if self.kind == "ppo":
            self.W1, self.b1 = w["pi.0.weight"], w["pi.0.bias"]
            self.W2, self.b2 = w["pi.2.weight"], w["pi.2.bias"]
            self.Wm, self.bm = w["action.weight"], w["action.bias"]
        else:
            self.W1, self.b1 = w["latent_pi.0.weight"], w["latent_pi.0.bias"]
            self.W2, self.b2 = w["latent_pi.2.weight"], w["latent_pi.2.bias"]
            self.Wm, self.bm = w["mu.weight"], w["mu.bias"]

    def __call__(self, obs):
        if self.kind == "ppo":
            h = np.tanh(self.W1 @ obs + self.b1)
            h = np.tanh(self.W2 @ h + self.b2)
            return float(np.clip((self.Wm @ h + self.bm)[0], -1.0, 1.0))
        h = np.maximum(0.0, self.W1 @ obs + self.b1)
        h = np.maximum(0.0, self.W2 @ h + self.b2)
        return float(np.tanh((self.Wm @ h + self.bm)[0]))
