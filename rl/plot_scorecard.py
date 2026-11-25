"""알고리즘별 강점·약점 한 장 + 후보군 분류.

    python rl/plot_scorecard.py

왼쪽은 성적표입니다. 한 줄을 가로로 읽으면 그 알고리즘이 **어디서 강하고 어디서 무너지는지**
모양이 보입니다. 숫자를 칸마다 적었고 색은 7개 안에서의 상대 위치일 뿐입니다(절대 기준 아님).
오른쪽은 교육용으로 쓸 수 있는지를 가르는 두 축 — 시간과 재현성 — 만 남긴 그림입니다.

색은 파랑↔주황 발산 배색입니다. 빨강-초록을 쓰지 않는 이유는 적록색약에서 두 끝이
구별되지 않기 때문이고, 어차피 칸마다 숫자가 있어 색은 보조 수단입니다.
"""
import os
import json
import io
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#ffffff"

BAD = np.array([247, 201, 174]) / 255.0
MID = np.array([240, 239, 234]) / 255.0
GOOD = np.array([187, 215, 242]) / 255.0

TIER_C = {"a": "#2a78d6", "b": "#eda100", "c": "#e34948"}

SLOTS = {"sac": "#1baf7a", "tqc": "#008300", "crossq": "#6250d6", "ppo": "#2a78d6",
         "ddpg": "#e87ba4", "td3": "#eda100", "a2c": "#eb6834"}
PRETTY = {"sac": "SAC", "tqc": "TQC", "crossq": "CrossQ", "ppo": "PPO",
          "ddpg": "DDPG", "td3": "TD3", "a2c": "A2C"}

ORDER = ["sac", "tqc", "crossq", "ppo", "a2c", "td3", "ddpg"]
TIER = {"sac": "a", "tqc": "a", "crossq": "a", "ppo": "a",
        "a2c": "c", "td3": "c", "ddpg": "c"}

KO = dict(
    title="교육용 키트에 쓸 강화학습 알고리즘: 강점과 약점",
    sub="스윙업+밸런싱 · 180초 예산 · 시드 4개 · 깨끗한 2차 측정(calib 23~30 ms) · 평가 시간은 시계에서 제외",
    p1="성적표 — 한 줄을 가로로 읽으면 그 알고리즘의 모양이 보입니다   ·   왼쪽 띠: 파랑 = 쓸 수 있음, 빨강 = 쓰면 안 됨",
    p2="쓸 수 있나 없나는 이 두 축이 가릅니다",
    x2="최초 성공까지 걸린 시간 [s]",
    y2="시드 편차  (시드를 바꿔도 같은 결과인가)",
    zone="교육용으로 쓸 수 있는 영역",
    scale_good="좋음", scale_bad="나쁨",
    note="색은 7개 안에서의 상대 위치입니다(절대 기준 아님) · 시드 편차 = 시간에 걸친 시드 간 성공률 최대-최소의 평균"
         " · 낙폭 = 리턴이 그때까지의 최고치에서 떨어진 최대폭",
    tier_a="쓸 수 있음", tier_c="쓰면 안 됨",
)
EN = dict(
    title="Choosing an RL algorithm for the teaching kit: strengths and weaknesses",
    sub="swing-up + balance | 180 s budget | 4 seeds | clean second repeat | evaluation time excluded",
    p1="scorecard - read a row across to see the shape   ·   left bar: blue = usable, red = not usable",
    p2="these two axes decide whether it is usable",
    x2="time to first success [s]",
    y2="seed spread (same result for everyone?)",
    zone="usable for teaching",
    scale_good="better", scale_bad="worse",
    note="colour is relative rank among the seven, not an absolute scale",
    tier_a="usable", tier_c="not usable",
)

COLS = [
    ("t_total_med", "최초 성공\n[s]", "time to\nsolve [s]", False, "%.0f"),
    ("seed_spread", "시드 편차\n(작을수록 좋음)", "seed\nspread", False, "%.3f"),
    ("drawdown_pct", "학습 중 낙폭\n[%]", "drawdown\n[%]", False, "%.0f"),
    ("final_success", "최종 성공률", "final\nsuccess", True, "%.2f"),
    ("final_ret", "최종 리턴", "final\nreturn", True, "%.0f"),
]


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic", "Noto Sans KR"]:
        if name in have:
            return name, KO
    return None, EN


def ramp(t):
    """t in [0,1], 0 = 나쁨, 1 = 좋음."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        f = t / 0.5
        c = BAD + (MID - BAD) * f
    else:
        f = (t - 0.5) / 0.5
        c = MID + (GOOD - MID) * f
    return tuple(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=os.path.join(HERE, "figures", "furuta_rl_wallclock_B_summary.json"))
    p.add_argument("--out", default=os.path.join(HERE, "figures"))
    p.add_argument("--name", default="furuta_rl_scorecard")
    a = p.parse_args()

    font, L = pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3, "font.size": 10,
    })

    M = json.load(io.open(a.summary, encoding="utf-8"))["metrics"]
    algos = [k for k in ORDER if k in M]

    fig = plt.figure(figsize=(15.2, 7.4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], left=0.105, right=0.975,
                          top=0.785, bottom=0.115, wspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ── 왼쪽: 성적표 ──
    ncol = len(COLS)
    ax1.set_xlim(0, ncol)
    ax1.set_ylim(len(algos) + 0.95, -1.28)
    ax1.axis("off")

    vals = {}
    for ci, (key, _ko, _en, higher, _f) in enumerate(COLS):
        col = np.array([M[k][key] for k in algos], dtype=float)
        lo, hi = col.min(), col.max()
        span = hi - lo if hi > lo else 1.0
        norm = (col - lo) / span
        if not higher:
            norm = 1.0 - norm
        vals[key] = (col, norm)

    for ri, k in enumerate(algos):
        for ci, (key, ko, en, higher, fmt) in enumerate(COLS):
            col, norm = vals[key]
            ax1.add_patch(Rectangle((ci + 0.035, ri + 0.07), 0.93, 0.86,
                                    facecolor=ramp(norm[ri]), edgecolor=SURFACE, lw=1.6))
            ax1.text(ci + 0.5, ri + 0.5, fmt % col[ri], ha="center", va="center",
                     fontsize=13, fontweight="bold", color=INK)
        ax1.text(-0.155, ri + 0.5, PRETTY[k], ha="right", va="center",
                 fontsize=13.5, fontweight="bold", color=INK)
        ax1.add_patch(Rectangle((-0.115, ri + 0.07), 0.038, 0.86,
                                facecolor=TIER_C[TIER[k]], edgecolor="none",
                                transform=ax1.transData, clip_on=False))

    for ci, (key, ko, en, higher, fmt) in enumerate(COLS):
        ax1.text(ci + 0.5, -0.16, ko if font else en, ha="center", va="bottom",
                 fontsize=10.5, color=INK2, linespacing=1.45)

    nA = sum(1 for k in algos if TIER[k] == "a")
    ax1.plot([-0.115, ncol], [nA, nA], color=INK3, lw=1.2, clip_on=False)

    gx0, gy = ncol - 1.35, len(algos) + 0.42
    for i in range(44):
        ax1.add_patch(Rectangle((gx0 + i * 0.0255, gy), 0.027, 0.19,
                                facecolor=ramp(i / 43.0), edgecolor="none", clip_on=False))
    ax1.text(gx0 - 0.07, gy + 0.095, L["scale_bad"], ha="right", va="center",
             fontsize=9, color=INK3, clip_on=False)
    ax1.text(gx0 + 44 * 0.0255 + 0.07, gy + 0.095, L["scale_good"], ha="left", va="center",
             fontsize=9, color=INK3, clip_on=False)
    ax1.text(-0.70, -1.08, L["p1"], ha="left", va="center", fontsize=10.5, color=INK2)

    # ── 오른쪽: 두 축 ──
    xs = [M[k]["t_total_med"] for k in algos]
    ys = [M[k]["seed_spread"] for k in algos]
    xhi = max(xs) * 1.16
    yhi = max(ys) * 1.16

    ax2.add_patch(Rectangle((0.5, 0.004), 90, 0.148,
                            facecolor="#1baf7a", alpha=0.085, edgecolor="none", zorder=0))
    ax2.add_patch(Rectangle((0.5, 0.004), 90, 0.148, facecolor="none",
                            edgecolor="#1baf7a", lw=1.1, ls=(0, (5, 3)), zorder=1))
    ax2.text(90, 0.160, L["zone"], ha="right", va="bottom", fontsize=9.5,
             color="#0d7a56", fontweight="bold")

    for k in algos:
        x, y = M[k]["t_total_med"], M[k]["seed_spread"]
        ax2.plot([x], [y], marker="o", ms=13, mfc=SLOTS[k], mec=SURFACE, mew=2.0, zorder=4)
        off = {"sac": (15, -1), "tqc": (-15, 9), "crossq": (15, 2), "ppo": (15, -1),
               "a2c": (-15, 0), "td3": (15, 0), "ddpg": (-15, 0)}.get(k, (15, 0))
        ax2.annotate(PRETTY[k], (x, y), xytext=off, textcoords="offset points",
                     ha="left" if off[0] > 0 else "right", va="center",
                     fontsize=12, fontweight="bold", color=INK, zorder=5)

    ax2.plot([0.06], [0.0], marker="D", ms=10, mfc=SURFACE, mec=INK2, mew=1.8, zorder=4)
    ax2.annotate("LQR+에너지 — 학습 0초\n단, 밀면 복원 63%" if font else "LQR+energy - no training\nbut only 63% recovery",
                 (0.06, 0.0), xytext=(24, 96), textcoords="offset points",
                 fontsize=9.5, color=INK2, zorder=5, linespacing=1.45,
                 arrowprops=dict(arrowstyle="-", color=INK3, lw=1.0,
                                 shrinkA=2, shrinkB=6, connectionstyle="arc3,rad=0.2"))

    ax2.set_xlim(-xhi * 0.035, xhi)
    ax2.set_ylim(-yhi * 0.035, yhi)
    ax2.set_xlabel(L["x2"], fontsize=10.5)
    ax2.set_ylabel(L["y2"], fontsize=10.5)
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_title(L["p2"], fontsize=11, color=INK2, loc="left", pad=10)

    fig.suptitle(L["title"], x=0.035, ha="left", fontsize=16, fontweight="bold", y=0.962)
    fig.text(0.035, 0.888, L["sub"], ha="left", fontsize=10, color=INK2)
    fig.text(0.035, 0.025, L["note"], ha="left", fontsize=8.5, color=INK3)

    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, a.name + ".png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(a.out, a.name + ".svg"))
    plt.close(fig)
    print("wrote", png, "(font: %s)" % (font or "default/EN"))

    for k in algos:
        print("%-8s %s" % (PRETTY[k], "  ".join(
            (fmt % M[k][key]).rjust(7) for key, _a, _b, _c, fmt in COLS)))


if __name__ == "__main__":
    main()
