"""1단계 시뮬 예선 결과를 한 장으로.

    python s2r/plot_stage1.py

왼쪽: 학습 시간에 따른 공칭 조건 엄격 통과 수(16칸). 선 = 시드, 점선 = 통과선 12/16.
오른쪽: 예산 끝 모델의 128칸 최종 평가를 조건별로. 회색 선 = 기준선(원래 정책 + 브리지 보상, 115/128).
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs_stage1")
OUT = os.path.join(HERE, "figures")

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de"
COL = {"sac": "#2a78d6", "ppo": "#eb6834"}
DASH = {0: None, 1: (5, 2.2), 2: (2, 1.7)}
LABEL = {"sac": "SAC", "ppo": "PPO"}
CONDS = ["nom", "J_lo", "J_hi", "fric_lo", "fric_hi", "k_lo", "k_hi", "kt_lat"]
CLAB = {"nom": "공칭", "J_lo": "관성 -", "J_hi": "관성 +", "fric_lo": "마찰 -", "fric_hi": "마찰 +",
        "k_lo": "결합 0.80", "k_hi": "결합 0.95", "kt_lat": "토크 0.9\n+지연"}


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if n in have:
            return n
    return None


def main():
    f = font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK3, "ytick.color": INK3, "font.size": 10})
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(RUNS, "*_s*.json")))]
    if not rows:
        print("결과 없음:", RUNS)
        return
    base = None
    bp = os.path.join(HERE, "baseline_eval.json")
    if os.path.exists(bp):
        base = json.load(open(bp, encoding="utf-8")).get("orig_bridge")

    fig = plt.figure(figsize=(15.0, 7.4), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], left=0.06, right=0.985, top=0.81, bottom=0.12,
                          wspace=0.18, hspace=0.42)
    axs = {"sac": fig.add_subplot(gs[0, 0]), "ppo": fig.add_subplot(gs[1, 0])}
    ax2 = fig.add_subplot(gs[:, 1])
    tmax = max(r["budget"] for r in rows) / 60.0
    for algo, ax in axs.items():
        for r in [r for r in rows if r["algo"] == algo]:
            t = np.array([p["t"] for p in r["points"]]) / 60.0
            v = np.array([p["strict"] for p in r["points"]])
            ln, = ax.plot(t, v, color=COL[algo], lw=1.9, drawstyle="steps-post", label="시드 %d" % r["seed"])
            if DASH[r["seed"] % 3]:
                ln.set_dashes(DASH[r["seed"] % 3])
            if r["t_pass"] is not None:
                ax.plot([r["t_pass"] / 60.0], [12], ls="none", marker="o", ms=6, mfc="white",
                        mec=COL[algo], mew=1.6, zorder=5)
        ax.axhline(12, color=INK3, lw=1.0, ls=(0, (2, 3)))
        ax.set_xlim(0, tmax)
        ax.set_ylim(-0.5, 16.8)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.set_ylabel("공칭 엄격 [/16]")
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="center right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5,
                  handlelength=2.4)
        ax.set_title("%s — 학습 중 성능 (1분마다, 점선 = 통과선 12/16)" % LABEL[algo], loc="left",
                     fontsize=11, fontweight="bold", color=COL[algo])
    axs["ppo"].set_xlabel("학습 벽시계 시간 [분]  (평가 시간 제외)")

    x = np.arange(len(CONDS))
    w = 0.38
    for i, algo in enumerate(("sac", "ppo")):
        rs = [r for r in rows if r["algo"] == algo]
        if not rs:
            continue
        v = np.array([[r["final_full"][c]["strict"] for c in CONDS] for r in rs])
        ax2.bar(x + (i - 0.5) * w, v.mean(0), width=w * 0.92, color=COL[algo], alpha=0.85,
                label="%s (합계 %s /128)" % (LABEL[algo], " · ".join(str(r["final_full"]["total_strict"]) for r in rs)))
        for j in range(len(CONDS)):
            ax2.plot([x[j] + (i - 0.5) * w] * len(rs), v[:, j], ls="none", marker="o", ms=4,
                     mfc="white", mec=INK2, mew=1.0, zorder=5)
    if base:
        for j, c in enumerate(CONDS):
            ax2.plot([x[j] - w, x[j] + w], [base[c]["strict"]] * 2, color=INK2, lw=1.6, zorder=6)
        ax2.plot([], [], color=INK2, lw=1.6, label="기준선: 원래 정책 + 브리지 보상 (%d/128)" % base["total_strict"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([CLAB[c] for c in CONDS], fontsize=9)
    ax2.set_ylim(0, 21.5)
    ax2.set_yticks([0, 4, 8, 12, 16])
    ax2.set_ylabel("엄격 통과 [/16]")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=1)
    ax2.set_title("예산 끝 모델의 최종 평가 (조건 8개 x 시작 16개)", loc="left", fontsize=11.5,
                  fontweight="bold", color=INK)

    b = rows[0]["budget"] / 60.0
    fig.suptitle("1단계 시뮬 예선: rev11 실기 보정 환경에서 PPO 대 SAC", x=0.055, ha="left", fontsize=15,
                 fontweight="bold", y=0.965)
    fig.text(0.055, 0.885, "알고리즘마다 시드 3개, 각 %.0f분 · 엄격 기준 = 12 s 안에 |직립 오차| ≤ 5°, |진자 각속도| ≤ 0.5 rad/s 를 5 s 연속 · "
             "환경 = s2r/s2r_env.py (마찰·관성·결합·토크·지연 무작위화)" % b, ha="left", fontsize=9.5, color=INK2)
    fig.text(0.055, 0.03, "SAC 3개는 동시에(프로세스당 코어 1개), PPO 는 하나씩(워커 24개) 돌렸습니다 · 점 = 처음 통과한 시각(왼쪽), 개별 시드(오른쪽)",
             ha="left", fontsize=8.5, color=INK3)
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "stage1_ppo_vs_sac.png")
    fig.savefig(png)
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
