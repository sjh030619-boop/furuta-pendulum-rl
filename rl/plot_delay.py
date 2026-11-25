"""지연 내성 실험 결과를 한 장으로.

    python rl/plot_delay.py

왼쪽·가운데: 구동 지연에 따른 성공률. 관측에 행동 이력이 없을 때와 있을 때를 나란히 두고,
선은 학습 때 지연을 어떻게 줬는지로 나눕니다. 센서는 실기 조건(12비트 + 차분 속도).
오른쪽: 스텝마다 지연이 달라지는 USB 루프를 가정했을 때의 성공률.

색은 `scratchpad/cvd.py` 로 확인한 조합입니다. 선 모양도 다르게 두었습니다.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"

TRAIN = [
    ("none", "학습 때 지연 없음", "#52514e", (5, 2.2), "o"),
    ("d20", "20 ms 고정으로 학습", "#eb6834", (2, 1.7), "s"),
    ("rand", "0~40 ms 무작위로 학습", "#2a78d6", None, "D"),
]
MODE = {
    (0, "none"): "plain", (0, "d20"): "d20", (0, "rand"): "rand",
    (2, "none"): "hist", (2, "d20"): "hist_d20", (2, "rand"): "hist_rand",
}
BAR_LABEL = {
    "plain": "이력 없음 · 지연 없이 학습",
    "d20": "이력 없음 · 20 ms 고정",
    "rand": "이력 없음 · 0~40 ms 무작위",
    "hist": "이력 있음 · 지연 없이 학습",
    "hist_d20": "이력 있음 · 20 ms 고정",
    "hist_rand": "이력 있음 · 0~40 ms 무작위",
}


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if n in have:
            return n
    return None


def frame(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default=os.path.join(HERE, "delay_eval.json"))
    p.add_argument("--sensor", default="real", choices=["real", "ideal"])
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_delay")
    a = p.parse_args()

    font = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    d = json.load(open(a.eval, encoding="utf-8"))
    delays = d["delays"]
    res = d["results"]
    usb = d["usb"]

    def group(mode):
        return [v for k, v in res.items() if v.get("mode") == mode]

    def curve(vs, key):
        return np.array([[v[key][str(ms)] for ms in delays] for v in vs])

    fig = plt.figure(figsize=(15.4, 6.6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], left=0.05, right=0.985,
                          top=0.76, bottom=0.17, wspace=0.14)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    lqr = res.get("LQR+energy")
    lo_ms, hi_ms = usb["delay_ms"], usb["delay_ms"] + usb["jitter_ms"]

    for ax, hist, title in ((axes[0], 0, "관측에 행동 이력 없음 (6차원)"),
                            (axes[1], 2, "관측에 직전 명령 2개 추가 (8차원)")):
        ax.axvspan(lo_ms, hi_ms, color=INK3, alpha=0.13, lw=0, zorder=1)
        ax.annotate("USB 노트북 루프\n%.0f~%.0f ms" % (lo_ms, hi_ms), ((lo_ms + hi_ms) / 2, 0.5),
                    ha="center", va="center", fontsize=8.5, color=INK2, linespacing=1.4)
        if lqr:
            y = [lqr[a.sensor][str(ms)] for ms in delays]
            ax.plot(delays, y, color=INK3, lw=1.4, ls=(0, (1, 2.2)), zorder=2,
                    label="LQR + 에너지 제어기")
        for key, lab, col, dash, mk in TRAIN:
            vs = group(MODE[(hist, key)])
            if not vs:
                continue
            c = curve(vs, a.sensor)
            ax.fill_between(delays, c.min(axis=0), c.max(axis=0), color=col, alpha=0.13,
                            lw=0, zorder=2)
            ln, = ax.plot(delays, c.mean(axis=0), color=col, lw=2.3, marker=mk, ms=5.5,
                          mfc=SURFACE, mec=col, mew=1.4, zorder=4, label=lab,
                          solid_capstyle="round")
            if dash:
                ln.set_dashes(dash)
        ax.set_xlim(-1.5, max(delays) + 1.5)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks(delays)
        ax.set_xlabel("구동 지연 [ms]  (제어 주기 20 ms)", fontsize=10)
        frame(ax)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
    axes[0].set_ylabel("성공률 (초기조건 15개)", fontsize=10)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=4, frameon=False,
                   fontsize=9.5, handlelength=2.8, columnspacing=1.6, borderaxespad=0.0)

    ax = axes[2]
    rows = []
    for hist in (0, 2):
        for key, _lab, col, _dash, _mk in TRAIN:
            m = MODE[(hist, key)]
            vs = group(m)
            if vs:
                rows.append((BAR_LABEL[m], [v["usb"] for v in vs], col, hist))
    if lqr:
        rows.append(("LQR + 에너지 제어기", [lqr["usb"]], INK3, 0))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, vals, col, hist) in zip(ys, rows):
        m = float(np.mean(vals))
        ax.barh(y, m, height=0.56, color=col if hist else SURFACE, edgecolor=col, lw=1.6,
                alpha=0.9 if hist else 1.0, zorder=3)
        if len(vals) > 1:
            ax.plot(vals, [y] * len(vals), ls="none", marker="o", ms=5, mfc=SURFACE,
                    mec=INK2, mew=1.2, zorder=5)
        ax.annotate("%.0f%%" % (100 * m), (max(m, max(vals)), y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK,
                    fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("성공률", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.tick_right()
    ax.set_title("지연이 매 스텝 %.0f~%.0f ms 로 흔들릴 때" % (lo_ms, hi_ms), fontsize=11.5,
                 color=INK, loc="left", pad=10, fontweight="bold")

    sensor = "실기 조건 센서(12비트 양자화 + 차분 속도)" if a.sensor == "real" else "이상적인 센서"
    fig.suptitle("구동 지연에 견디는 SAC: 무엇을 바꿔야 하나", x=0.05, ha="left", fontsize=15,
                 fontweight="bold", y=0.965)
    fig.text(0.05, 0.895,
             "SAC (학습률 1e-3, 전이 20만) · %s · 시드 3개, 띠 = 시드 최소~최대 · "
             "성공 = 5초 에피소드의 마지막 2초를 도립으로 유지" % sensor,
             ha="left", fontsize=9.5, color=INK2)
    fig.text(0.05, 0.03,
             "지연은 2 ms 단위로 줍니다(`rl/delay_env.py`) · 오른쪽 칸: 빈 막대 = 이력 없음, 채운 막대 = 이력 있음, 점 = 개별 시드",
             ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
