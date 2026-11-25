"""학습 곡선: 시간(그리고 경험량)에 따라 128칸 성능이 어떻게 오르나, 알고리즘별 색 + 시드 편차 띠.

    python s2r/plot_curves.py

데이터: 1단계(각 25분) 체크포인트를 1분마다 128칸으로 다시 평가한 값 (runs_stage4/ckpt_{sac,ppo}_s*.json).
선 = 시드 3개 평균, 연한 띠 = 시드 3개의 최저~최고.
왼쪽: x = 학습 벽시계 시간(분). 오른쪽: x = 모은 경험(환경 스텝, 로그 눈금).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de"
COL = {"sac": "#2a78d6", "ppo": "#eb6834"}
LAB = {"sac": "SAC (off-policy, 경험 재사용)", "ppo": "PPO (on-policy, 경험 한 번 쓰고 버림)"}


def J(p):
    return json.load(open(p, encoding="utf-8"))


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if n in have:
            return n
    return None


def load(algo):
    T, S, Y = [], [], []
    for s in (0, 1, 2):
        ck = J(os.path.join(HERE, "runs_stage4", "ckpt_%s_s%d.json" % (algo, s)))
        pts = J(os.path.join(HERE, "runs_stage1", "%s_s%d.json" % (algo, s)))["points"]
        t = np.array([c["t"] for c in ck], dtype=float)
        y = np.array([c["strict"] for c in ck], dtype=float)
        pt = np.array([p["t"] for p in pts])
        ps = np.array([p["steps"] for p in pts], dtype=float)
        steps = np.array([ps[np.argmin(np.abs(pt - x))] for x in t])
        T.append(t / 60.0)
        S.append(steps)
        Y.append(y)
    n = min(len(y) for y in Y)
    return np.array([t[:n] for t in T]), np.array([s[:n] for s in S]), np.array([y[:n] for y in Y])


def main():
    f = font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK3, "ytick.color": INK3, "font.size": 10.5})
    data = {a: load(a) for a in ("sac", "ppo")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.6), dpi=150)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.2, wspace=0.16)
    for a in ("sac", "ppo"):
        T, S, Y = data[a]
        t = T.mean(0)
        ax1.fill_between(t, Y.min(0), Y.max(0), color=COL[a], alpha=0.16, lw=0)
        ax1.plot(t, Y.mean(0), color=COL[a], lw=2.6, label=LAB[a])
        s = np.exp(np.log(np.maximum(S, 1)).mean(0))
        ok = s > 0
        ax2.fill_between(s[ok], Y.min(0)[ok], Y.max(0)[ok], color=COL[a], alpha=0.16, lw=0)
        ax2.plot(s[ok], Y.mean(0)[ok], color=COL[a], lw=2.6, label=LAB[a])
    for ax in (ax1, ax2):
        ax.axhline(120, color=INK3, lw=1.0, ls=(0, (3, 3)))
        ax.set_ylim(-4, 134)
        ax.set_yticks([0, 32, 64, 96, 128])
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.text(24.7, 117, "바탕 정책 기준 120", ha="right", va="top", fontsize=9, color=INK3)
    ax1.set_xlim(0, 25)
    ax1.set_xlabel("학습 시간 [분]  (벽시계, 평가 시간 제외)")
    ax1.set_ylabel("128칸 시험 통과 수")
    ax1.set_title("시간에 따른 성능", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax2.set_xscale("log")
    ax2.set_xlim(2e4, 2e7)
    ax2.set_xlabel("모은 경험 [환경 스텝, 로그 눈금]")
    ax2.set_title("경험량에 따른 성능 (같은 경험으로 얼마나 배우나)", loc="left", fontsize=13, fontweight="bold", color=INK)

    fig.suptitle("학습 곡선: SAC 대 PPO (rev11 실기 보정 시뮬)", x=0.045, ha="left", fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.045, 0.9, "선 = 시드 3개 평균, 연한 띠 = 시드 3개의 최저~최고 · 1분마다 저장한 정책을 128칸(조건 8개 x 시작 16개)으로 평가 · "
             "점선 = 바탕 정책 기준 120", ha="left", fontsize=10, color=INK2)
    h = [plt.Line2D([], [], color=COL[a], lw=2.6) for a in ("sac", "ppo")]
    fig.legend(h, [LAB["sac"], LAB["ppo"]], loc="upper left", bbox_to_anchor=(0.04, 0.855), ncol=2, frameon=False,
               fontsize=11)
    notes = [("sac", "SAC: 2분에 평균 100을 넘음. 다만 시드마다 한 번씩 잠깐 무너졌다 회복(띠가 아래로 벌어지는 곳) → 좋은 순간에 멈추는 규칙으로 해결"),
             ("ppo", "PPO: 시드마다 크게 다름(5분에 오른 시드, 16분에 오른 시드, 25분까지 못 오른 시드). 경험은 SAC 의 약 8배를 모았지만 늦게 오름")]
    for k, (a, t) in enumerate(notes):
        fig.text(0.045, 0.085 - 0.04 * k, t, ha="left", fontsize=10, color=COL[a])
    fig.text(0.985, 0.012, "SAC 는 3개를 동시에 돌려 혼자 돌릴 때보다 약 25% 느림(시간 축에서 SAC 에 불리) · 데이터: runs_stage4/ckpt_*.json",
             ha="right", fontsize=8.5, color=INK3)
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "learning_curves.png")
    fig.savefig(png)
    plt.close(fig)
    for a in ("sac", "ppo"):
        T, S, Y = data[a]
        print(a, "평균 곡선:", " ".join("%d" % v for v in Y.mean(0)), "| 25분 경험", int(S[:, -1].mean()))
    print("wrote", png)


if __name__ == "__main__":
    main()
