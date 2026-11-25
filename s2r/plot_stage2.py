"""2단계 실기 본선 결과를 한 장으로.

    python s2r/plot_stage2.py

위: 팔(정책)마다 5회의 진자 각도. 0° = 매달림, 180° = 직립. 점선 = 직립 ±5°.
아래 왼쪽: 직립 유지 시작 시각(엄격 기준 5 s 연속 구간의 시작), 점 = 각 회.
아래 가운데: 7~12 s 직립 오차 RMS.  아래 오른쪽: 7~12 s 평균 |iq| (전류, 발열·소음 지표).
기록: s2r/stage2/logs (무효 회는 하위 폴더로 빼 둠).
"""
import os
import csv
import glob
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "stage2", "logs")
OUT = os.path.join(HERE, "figures")
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de"
ARMS = ["base", "sac_s0", "sac_s1", "sac_s2", "ppo_s2"]
LAB = {"base": "기준선\n원래 정책+브리지 보상", "sac_s0": "SAC 시드 0", "sac_s1": "SAC 시드 1",
       "sac_s2": "SAC 시드 2", "ppo_s2": "PPO 시드 2 (참고)"}
COL = {"base": "#52514e", "sac_s0": "#2a78d6", "sac_s1": "#2a78d6", "sac_s2": "#2a78d6", "ppo_s2": "#eb6834"}


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if n in have:
            return n
    return None


def load(arm):
    out = []
    for p in sorted(glob.glob(os.path.join(LOGS, "r*__%s__*.csv" % arm))):
        if p.endswith(("_settle.csv", "_fric.csv")):
            continue
        r = list(csv.DictReader(open(p)))
        t = np.array([float(x["t"]) for x in r])
        t -= t[0]
        al = np.array([float(x["alpha"]) for x in r])
        iq = np.array([abs(int(x["iq"])) / 100.0 if x["iq"] else np.nan for x in r])
        deg = np.degrees(np.abs((al + np.pi) % (2 * np.pi) - np.pi))
        out.append((os.path.basename(p), t, deg, iq))
    return out


def main():
    f = font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK3, "ytick.color": INK3, "font.size": 10})
    summ = json.load(open(os.path.join(LOGS, "stage2_summary.json"), encoding="utf-8"))
    hold = {a: [r["t_hold"] for r in summ["runs"] if r["arm"] == a] for a in ARMS}
    data = {a: load(a) for a in ARMS}

    fig = plt.figure(figsize=(15.0, 8.6), dpi=150)
    gs = fig.add_gridspec(2, 5, left=0.085, right=0.985, top=0.83, bottom=0.11, hspace=0.55, wspace=0.28,
                          height_ratios=[1.0, 1.0])
    for j, a in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, j])
        for _n, t, deg, _iq in data[a]:
            ax.plot(t, deg, color=COL[a], lw=1.1, alpha=0.75)
        ax.axhspan(175, 180, color=GRID, zorder=0)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 190)
        ax.set_yticks([0, 90, 180])
        ax.set_xticks([0, 4, 8, 12])
        ax.grid(color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("%s\n엄격 %d/%d" % (LAB[a], summ["summary"][a]["strict"], summ["summary"][a]["n"]),
                     loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("시간 [s]")
        if j == 0:
            ax.set_ylabel("진자 각도 [°]\n(180 = 직립)")

    names = [LAB[a].split("\n")[0] for a in ARMS]
    y = np.arange(len(ARMS))[::-1]
    ax = fig.add_subplot(gs[1, 0:2])
    for k, a in enumerate(ARMS):
        v = np.array(hold[a], dtype=float)
        ax.plot([np.median(v)] * 2, [y[k] - 0.28, y[k] + 0.28], color=INK, lw=2.0)
        ax.plot(v, y[k] + np.linspace(-0.15, 0.15, len(v)), ls="none", marker="o", ms=7, mfc=COL[a],
                mec="white", mew=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("직립 유지 시작 [s]  (막대 = 중앙값)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("얼마나 빨리 세우나", loc="left", fontsize=11, fontweight="bold", color=INK)

    rms, cur = [], []
    for a in ARMS:
        e, c = [], []
        for _n, t, deg, iq in data[a]:
            m = t >= 7.0
            e.append(180.0 - deg[m])
            c.append(iq[m])
        e = np.concatenate(e)
        c = np.concatenate(c)
        rms.append(float(np.sqrt(np.mean(e ** 2))))
        cur.append(float(np.nanmean(c)))
    for col, vals, title, unit, lim in ((2, rms, "직립 오차 RMS (7~12 s)", "°", 2.0),
                                        (3, cur, "전류: 평균 |iq| (7~12 s)", " A", 0.7)):
        ax = fig.add_subplot(gs[1, col:col + 1] if col == 2 else gs[1, 3:5])
        ax.barh(y, vals, height=0.6, color=[COL[a] for a in ARMS], alpha=0.85)
        for k, v in enumerate(vals):
            ax.text(v + lim * 0.02, y[k], ("%.2f" % v) + unit, va="center", fontsize=9, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlim(0, lim)
        ax.grid(axis="x", color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)

    fig.suptitle("2단계 실기 본선: rev11 실기에서 정책마다 5회 (매달려 멈춘 상태에서 시작)", x=0.055, ha="left",
                 fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.055, 0.905, "엄격 기준 = 12 s 안에 |직립 오차| ≤ 5°, |진자 각속도| ≤ 0.5 rad/s 를 5 s 연속 · 1.2 A 상한 · 제동 전력 제한 5 W · "
             "SAC/PPO 는 학습 정책을 그대로(--tau 2.52), 기준선은 --tau 1.0 --comp", ha="left", fontsize=9.5, color=INK2)
    fig.text(0.055, 0.012, "2026-11-24 · 모터 온도 38→41°C · 시작 상태가 잘못된 회(흔들리며 시작 5회, 직립에서 시작 3회)는 빼고 다시 돌림",
             ha="left", fontsize=8.5, color=INK3)
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, "stage2_hardware.png")
    fig.savefig(png)
    plt.close(fig)
    print("wrote", png)
    print("RMS", dict(zip(ARMS, [round(v, 2) for v in rms])), "iq", dict(zip(ARMS, [round(v, 3) for v in cur])))


if __name__ == "__main__":
    main()
