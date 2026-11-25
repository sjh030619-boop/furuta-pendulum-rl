"""PPO 대 SAC 정리 그래프 (발표용).

    python s2r/plot_summary.py

figures/summary_tradeoff.png   관점 여섯 개에서 SAC 와 PPO 를 나란히 (점 = 시드 또는 바탕 정책 하나)
figures/summary_finetune.png   학생 보상 x 바탕 정책: 고쳐 배운 뒤 128칸 통과 수
숫자 요약은 summary_numbers.json 으로도 저장(발표자료 MD 가 인용).
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de"
COL = {"sac": "#2a78d6", "ppo": "#eb6834"}
LAB = {"sac": "SAC", "ppo": "PPO"}
PPO_PRE = 1500.0


def J(path):
    return json.load(open(path, encoding="utf-8"))


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if n in have:
            return n
    return None


def collect():
    d = {"sac": {}, "ppo": {}}
    d["sac"]["t_base_min"] = [J(os.path.join(HERE, "runs_stage4", "train_du0_s%d.json" % s))["t_stop"] / 60 for s in (0, 1, 2)]
    d["sac"]["g128"] = [J(os.path.join(HERE, "runs_stage4", "train_du0_s%d.json" % s))["full_strict"] for s in (0, 1, 2)]
    d["sac"]["ok"] = [True, True, True]
    t, g, ok = [], [], []
    for s in (0, 1, 2):
        r = J(os.path.join(HERE, "runs_base", "cont__ppo_a_s%d.json" % s))
        ok.append(r["stopped"] == "rule")
        t.append((PPO_PRE + r["t_stop"]) / 60)
        g.append(r["full_strict"])
    d["ppo"].update(t_base_min=t, g128=g, ok=ok)
    hw = J(os.path.join(HERE, "stage2", "logs", "stage2_summary.json"))["summary"]
    d["sac"]["hw_iq"] = [0.408, 0.573, 0.360]
    d["ppo"]["hw_iq"] = [0.196]
    d["sac"]["hw_t"] = [hw["sac_s%d" % s]["t_hold_med"] for s in (0, 1, 2)]
    d["ppo"]["hw_t"] = [hw["ppo_s2"]["t_hold_med"]]
    d["sac"]["hw_pass"] = "%d/15" % sum(hw["sac_s%d" % s]["strict"] for s in (0, 1, 2))
    d["ppo"]["hw_pass"] = "%d/5" % hw["ppo_s2"]["strict"]
    rows = [J(f) for f in sorted(glob.glob(os.path.join(HERE, "runs_reward", "base_*__*__*.json")))]
    for a in ("sac", "ppo"):
        bases = sorted({r["base"] for r in rows if r["base"].startswith("base_" + a)})
        ft_pass, ft_keep, n_good, n_bad, bad_rej, mins, iq = [], [], 0, 0, 0, [], []
        for b in bases:
            good = [r for r in rows if r["base"] == b and r["kind"] != "bad"]
            bad = [r for r in rows if r["base"] == b and r["kind"] == "bad"]
            passed = [r for r in good if not r["picked"].startswith("바탕")]
            ft_pass.append(100.0 * len(passed) / len(good))
            ft_keep.append(100.0 * sum(r["gate128"] >= 120 for r in passed) / max(len(good), 1))
            mins.append(min(r["gate128"] for r in passed) if passed else None)
            n_good += len(good)
            n_bad += len(bad)
            bad_rej += sum(r["picked"].startswith("바탕") for r in bad)
            iq.append(J(os.path.join(HERE, "runs_reward", "_base__%s.json" % b))["behavior"]["hold_iq"])
        d[a].update(bases=bases, ft_pass=ft_pass, ft_keep=ft_keep, ft_min=mins, n_good=n_good, n_bad=n_bad,
                    bad_rej=bad_rej, sim_iq=iq,
                    n_pass=sum(1 for r in rows if r["base"] in bases and r["kind"] != "bad" and not r["picked"].startswith("바탕")),
                    n_keep=sum(1 for r in rows if r["base"] in bases and r["kind"] != "bad"
                               and not r["picked"].startswith("바탕") and r["gate128"] >= 120))
    return d, rows


def tradeoff(d):
    fig, axs = plt.subplots(2, 3, figsize=(15.0, 8.2), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.07, hspace=0.75, wspace=0.32)
    panels = [
        ("① 바탕 정책 만드는 시간", "분 (짧을수록 좋음)", "t_base_min", (0, 50), "low"),
        ("② 시드마다 고르게 되나", "128칸 통과 수 (높을수록 좋음)", "g128", (0, 135), "high"),
        ("③ 실기에서 조용히 서 있나", "서 있을 때 전류 A (낮을수록 좋음)", "hw_iq", (0, 0.7), "low"),
        ("④ 실기에서 빨리 세우나", "직립 유지 시작 s (짧을수록 좋음)", "hw_t", (0, 3.2), "low"),
        ("⑤ 학생 보상으로 고쳐 배우기", "고정 시험 통과 % (높을수록 좋음)", "ft_pass", (0, 108), "high"),
        ("⑥ 고친 뒤에도 튼튼한가", "128칸 120 이상 유지 % (높을수록 좋음)", "ft_keep", (0, 108), "high"),
    ]
    for ax, (title, xl, key, lim, better) in zip(axs.ravel(), panels):
        med = {}
        for k, a in enumerate(("sac", "ppo")):
            y = 1 - k
            v = np.array(d[a][key], dtype=float)
            ok = np.array(d[a]["ok"]) if key in ("t_base_min", "g128") else np.ones(len(v), bool)
            jit = np.linspace(-0.13, 0.13, len(v)) if len(v) > 1 else np.zeros(1)
            if key == "t_base_min":
                ax.plot(v[ok], y + jit[ok], ls="none", marker="o", ms=11, mfc=COL[a], mec="white", mew=1.5)
                for j in np.flatnonzero(~ok):
                    ax.plot([lim[1] * 0.93], [y + jit[j]], ls="none", marker="x", ms=11, mec=COL[a], mew=2.6)
                    ax.text(lim[1] * 0.93, y + jit[j] + 0.2, "45분에도 실패", ha="center", fontsize=8.5, color=COL[a])
                med[a] = float(np.median(v[ok])) if ok.any() else None
            else:
                ax.plot(v, y + jit, ls="none", marker="o", ms=11, mfc=COL[a], mec="white", mew=1.5)
                med[a] = float(np.median(v))
            if med[a] is not None:
                ax.plot([med[a]] * 2, [y - 0.3, y + 0.3], color=INK, lw=2.2, zorder=1)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["SAC", "PPO"], fontsize=12, fontweight="bold")
        for tl, a in zip(ax.get_yticklabels(), ("sac", "ppo")):
            tl.set_color(COL[a])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(*lim)
        ax.set_xlabel(xl, fontsize=9.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=22)
        if med["sac"] is None or med["ppo"] is None:
            win = "sac" if med["sac"] is not None else "ppo"
        else:
            diff = abs(med["sac"] - med["ppo"]) / max(abs(med["sac"]), abs(med["ppo"]), 1e-9)
            if key == "g128" and min(d["ppo"][key]) < 60:
                win = "sac"
            elif diff < 0.05:
                win = None
            else:
                win = ("sac" if med["sac"] < med["ppo"] else "ppo") if better == "low" else \
                      ("sac" if med["sac"] > med["ppo"] else "ppo")
        ax.text(0.0, 1.06, ("→ %s 가 나음" % LAB[win]) if win else "→ 비슷함", transform=ax.transAxes, fontsize=10.5,
                color=COL[win] if win else INK2, fontweight="bold")
    fig.suptitle("PPO 대 SAC: 관점마다 누가 나은가", x=0.045, ha="left", fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.045, 0.895, "점 하나 = 시드(또는 바탕 정책) 하나, 검은 막대 = 중앙값 · rev11 실기 보정 시뮬 + 실기 · "
             "실기(③④)는 1단계 정책으로 측정, PPO 는 가장 좋은 시드 하나만", ha="left", fontsize=10, color=INK2)
    fig.savefig(os.path.join(OUT, "summary_tradeoff.png"))
    plt.close(fig)


def finetune(d, rows):
    from student_rewards import REWARDS
    bases = d["sac"]["bases"] + d["ppo"]["bases"]
    M = np.full((len(REWARDS), len(bases)), np.nan)
    txt = [["" for _ in bases] for _ in REWARDS]
    for r in rows:
        if r["base"] not in bases:
            continue
        i = [x["key"] for x in REWARDS].index(r["reward"])
        j = bases.index(r["base"])
        if r["picked"].startswith("바탕"):
            txt[i][j] = "걸러짐" if r["kind"] == "bad" else ("탈락" if "탈락" in r["picked"] else "개선 없음")
        else:
            M[i, j] = r["gate128"]
            txt[i][j] = "%d" % r["gate128"]
    cmap = LinearSegmentedColormap.from_list("g", ["#f3d9cf", "#f4f1e6", "#cfe6d5", "#3f9c63"])
    fig, ax = plt.subplots(figsize=(11.5, 8.4), dpi=150)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.05)
    ax.imshow(np.clip(M, 96, 128), cmap=cmap, vmin=96, vmax=128, aspect="auto")
    for i in range(len(REWARDS)):
        for j in range(len(bases)):
            miss = np.isnan(M[i, j])
            if miss:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#ecebe6", zorder=1))
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=10.5, zorder=2,
                    color=INK3 if miss else INK, fontweight="normal" if miss else "bold")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels(["%s 바탕 %s\n(128칸 %d)" % (LAB[b.split("_")[1]], b.split("_")[2][1:],
                                                 J(os.path.join(HERE, "runs_reward", "_base__%s.json" % b))["gate128"])
                        for b in bases], fontsize=10)
    ax.xaxis.tick_top()
    for tl, b in zip(ax.get_xticklabels(), bases):
        tl.set_color(COL[b.split("_")[1]])
        tl.set_fontweight("bold")
    kind = {"good": "", "hard": "  (어려움)", "bad": "  (잘못된 규칙)"}
    ax.set_yticks(range(len(REWARDS)))
    ax.set_yticklabels(["%s%s" % (r["name"], kind[r["kind"]]) for r in REWARDS], fontsize=10.5)
    ax.set_xticks(np.arange(-0.5, len(bases)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REWARDS)), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ns = len(d["sac"]["bases"])
    if d["ppo"]["bases"]:
        ax.axvline(ns - 0.5, color=INK2, lw=1.6)
    fig.suptitle("학생 보상으로 고쳐 배운 뒤: 128칸 시험 통과 수", x=0.03, ha="left", fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.03, 0.915, "앱 방식 CEM, 앱 기준 약 22초 · 숫자 = 고친 정책의 128칸 통과 수(진한 초록일수록 튼튼) · "
             "회색 = 고정 시험에서 떨어져 바탕 정책 유지", ha="left", fontsize=9.5, color=INK2)
    fig.savefig(os.path.join(OUT, "summary_finetune.png"))
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, HERE)
    f = font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK3, "ytick.color": INK3, "font.size": 10})
    os.makedirs(OUT, exist_ok=True)
    d, rows = collect()
    tradeoff(d)
    finetune(d, rows)
    with open(os.path.join(HERE, "summary_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
    for a in ("sac", "ppo"):
        x = d[a]
        print(a, "바탕 시간(분)", [round(v, 1) for v in x["t_base_min"]], "성공", x["ok"], "128칸", x["g128"],
              "| 고쳐 배우기 통과 %d/%d, 120 이상 %d, 최저 %s, 잘못된 규칙 걸러짐 %d/%d" % (
                  x["n_pass"], x["n_good"], x["n_keep"], x["ft_min"], x["bad_rej"], x["n_bad"]),
              "| 시뮬 전류", [round(v, 2) for v in x["sim_iq"]])


if __name__ == "__main__":
    main()
