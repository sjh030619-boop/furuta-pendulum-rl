"""rev11 정책의 스윙업을 GIF 로 그립니다.

    python rl/render_rev11.py                               # rl/rev11/sac_rev11.zip
    python rl/render_rev11.py --classical --al0 0.05        # 비교용: LQR + 에너지 제어기

위는 MuJoCo 화면, 아래는 진자 각도와 모터 토크 그래프입니다. 진자 각도는 "아래에서 몇 도
올라왔나"(0° = 매달림, 180° = 도립)로 그립니다. 어느 쪽으로 흔들리든 같은 값이라 스윙
진폭이 커지는 모습이 그대로 보입니다.

매달려 정지한 상태에서 시작하고, 정책은 결정적(deterministic)으로 돌립니다.
제어 50 Hz 를 두 스텝마다 한 장씩 그려 25 fps, 실시간 속도입니다.
"""
import os
import sys
import math
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco
from PIL import Image, ImageDraw, ImageFont

from train_rev11 import ENV_KW, OUT
from furuta_env import FurutaEnv

INK = (11, 11, 11)
INK2 = (82, 81, 78)
INK3 = (138, 136, 128)
GRID = (228, 227, 222)
BG = (255, 255, 255)
C_ANG = (42, 120, 214)
C_TAU = (235, 104, 52)
C_UP = (27, 175, 122)


def font(size, bold=False):
    for f in (("malgunbd.ttf" if bold else "malgun.ttf"), "NanumGothic.ttf", "arial.ttf"):
        for d in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/nanum", ""):
            p = os.path.join(d, f)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rollout(env, act, seconds, al0=0.0):
    """매달려 정지한 상태에서 시작해 매 제어 스텝의 상태와 실제 토크를 기록합니다."""
    obs = env.set_state(0.0, al0, 0.0, 0.0)
    n = int(round(seconds / env.ctrl_dt))
    qs, vs, taus, ups = [env.data.qpos.copy()], [env.data.qvel.copy()], [0.0], [0.0]
    for _ in range(n):
        u = act(obs)
        obs, _r, term, _tr, info = env.step(np.array([u], dtype=np.float32))
        qs.append(env.data.qpos.copy())
        vs.append(env.data.qvel.copy())
        taus.append(float(env.data.ctrl[0]))
        ups.append(info["upright"])
        if term:
            break
    return np.array(qs), np.array(vs), np.array(taus), np.array(ups)


def lift_deg(al):
    """아래에서 올라온 각도. 0 = 매달림, 180 = 도립. 흔들리는 방향과 무관."""
    w = (al + math.pi) % (2 * math.pi) - math.pi
    return np.degrees(np.abs(w))


def strip(w, h, t, ang, tau, k, tau_max, t_up, f_s, f_xs):
    """아래쪽 그래프. 지금 시각 k 까지만 그립니다."""
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    L, R = 58, w - 16
    a_top, a_bot = 26, 84
    t_top, t_bot = 112, 146
    axis_y = h - 18
    T_end = t[-1]

    def X(tt):
        return L + (R - L) * tt / T_end

    def YA(a):
        return a_bot - (a_bot - a_top) * a / 180.0

    def YT(v):
        mid = (t_top + t_bot) / 2
        return mid - (t_bot - t_top) / 2 * v / tau_max

    d.line([(0, 0), (w, 0)], fill=GRID, width=1)
    d.text((L, 6), "진자 각도", fill=C_ANG, font=f_s, anchor="lt")
    d.text((L + 64, 7), "아래 0° → 도립 180°", fill=INK3, font=f_xs, anchor="lt")
    d.text((L, 92), "모터 토크 [N·m]", fill=C_TAU, font=f_s, anchor="lt")
    for a in (0, 90):
        d.line([(L, YA(a)), (R, YA(a))], fill=GRID, width=1)
        d.text((L - 7, YA(a)), "%d°" % a, fill=INK3, font=f_xs, anchor="rm")
    d.line([(L, YA(180)), (R, YA(180))], fill=C_UP, width=1)
    d.text((L - 7, YA(180)), "180°", fill=C_UP, font=f_xs, anchor="rm")
    for v, lab in ((tau_max, "+%.0f" % tau_max), (0.0, "0"), (-tau_max, "-%.0f" % tau_max)):
        d.line([(L, YT(v)), (R, YT(v))], fill=GRID, width=1)
        d.text((L - 7, YT(v)), lab, fill=INK3, font=f_xs, anchor="rm")
    for s in range(0, int(math.floor(T_end)) + 1):
        d.line([(X(s), axis_y - 4), (X(s), axis_y)], fill=INK3, width=1)
        d.text((X(s), axis_y + 2), "%d s" % s, fill=INK3, font=f_xs, anchor="mt")

    if k >= 1:
        d.line([(X(t[i]), YA(ang[i])) for i in range(k + 1)], fill=C_ANG, width=2)
        d.line([(X(t[i]), YT(tau[i])) for i in range(k + 1)], fill=C_TAU, width=2)
    if t_up is not None and t[k] >= t_up:
        d.line([(X(t_up), a_top - 2), (X(t_up), t_bot)], fill=C_UP, width=1)
    d.line([(X(t[k]), a_top - 2), (X(t[k]), t_bot)], fill=INK2, width=1)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=os.path.join(OUT, "sac_rev11.zip"))
    p.add_argument("--classical", action="store_true")
    p.add_argument("--out", default=os.path.join(OUT, "rev11_swingup.gif"))
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=440)
    p.add_argument("--strip", type=int, default=172)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--azimuth", type=float, default=90.0,
                   help="90 = 장치 앞면을 정면으로. 처음 매달린 자세와 마지막 도립 자세가 둘 다 안 가려짐")
    p.add_argument("--elevation", type=float, default=-20.0)
    p.add_argument("--distance", type=float, default=0.50)
    p.add_argument("--lookat", type=float, nargs=3, default=[-0.04, 0.0, -0.015],
                   help="x 를 음수로 두면 장치가 화면 오른쪽으로 가서 왼쪽 위 글자와 안 겹침")
    p.add_argument("--sheet", default="")
    p.add_argument("--delay-ms", type=float, default=0.0,
                   help="구동 지연 [ms]. 지연 대응 정책을 지연 조건에서 보여줄 때")
    p.add_argument("--hist", type=int, default=-1,
                   help="관측의 직전 명령 개수. -1 이면 정책 파일 이름으로 판단(delay 면 2)")
    p.add_argument("--al0", type=float, default=0.0,
                   help="시작 진자각 [rad]. 고전 제어기는 정확히 0 에서 sign(0)=0 이라 움직이지 않음")
    a = p.parse_args()

    hist = a.hist if a.hist >= 0 else (2 if "delay" in os.path.basename(a.policy) else 0)
    if a.classical:
        hist = 0
    if hist or a.delay_ms:
        from delay_env import DelayEnv
        env = DelayEnv(delay_ms=a.delay_ms, hist=hist, episode_sec=a.seconds + 1.0, **ENV_KW)
    else:
        env = FurutaEnv(episode_sec=a.seconds + 1.0, **ENV_KW)
    env.reset(seed=0)

    if a.classical:
        from robust import ClassicalPolicy
        cp = ClassicalPolicy(FurutaEnv(**ENV_KW))

        def act(o):
            return float(cp.predict(o[None, :])[0][0, 0])
        name = "LQR + 에너지 스윙업"
    else:
        from stable_baselines3 import SAC
        model = SAC.load(a.policy, device="cpu")

        def act(o):
            return float(model.predict(o, deterministic=True)[0][0])
        name = "SAC 지연 대응 정책" if hist else "SAC 정책 스윙업"
    if a.delay_ms:
        name += " · 지연 %.0f ms" % a.delay_ms

    qs, vs, taus, ups = rollout(env, act, a.seconds, a.al0)
    dt = env.ctrl_dt
    t = np.arange(len(qs)) * dt
    ang = lift_deg(qs[:, 1])
    hold = int(round(0.5 / dt))
    t_up = None
    for i in range(len(ups) - hold):
        if ups[i:i + hold].min() > 0.5:
            t_up = float(t[i])
            break
    tail = int(round(2.0 / dt))
    stats = dict(t_up=t_up, final_upright=float(ups[-tail:].mean()),
                 tau_peak=float(np.abs(taus).max()),
                 arm_range_deg=[float(np.degrees(qs[:, 0].min())), float(np.degrees(qs[:, 0].max()))],
                 w_arm_peak=float(np.abs(vs[:, 0]).max()),
                 w_pend_peak=float(np.abs(vs[:, 1]).max()),
                 swings=int(np.sum(np.diff(np.sign(vs[:, 1])) != 0)))
    print("도립 도달(0.5 s 유지 시작) %s, 마지막 2초 도립 비율 %.2f, 토크 피크 %.2f N m, "
          "팔 각도 %.0f°~%.0f°" % ("%.2f s" % t_up if t_up is not None else "없음",
                                  stats["final_upright"], stats["tau_peak"],
                                  stats["arm_range_deg"][0], stats["arm_range_deg"][1]))

    r = mujoco.Renderer(env.model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = a.lookat
    cam.distance = a.distance
    cam.azimuth = a.azimuth
    cam.elevation = a.elevation

    f_t = font(18, True)
    f_sub = font(12)
    f_m = font(15)
    f_s = font(13)
    f_xs = font(11)
    data = env.data
    frames, durs = [], []
    idx = list(range(0, len(qs), a.every))
    if idx[-1] != len(qs) - 1:
        idx.append(len(qs) - 1)
    for k in idx:
        data.qpos[:] = qs[k]
        data.qvel[:] = vs[k]
        mujoco.mj_forward(env.model, data)
        r.update_scene(data, camera=cam)
        canvas = Image.new("RGB", (a.width, a.height + a.strip), BG)
        canvas.paste(Image.fromarray(r.render()), (0, 0))
        canvas.paste(strip(a.width, a.strip, t, ang, taus, k, env.tau_max, t_up, f_s, f_xs),
                     (0, a.height))
        d = ImageDraw.Draw(canvas)
        d.text((14, 10), "rev11 · %s" % name, fill=INK, font=f_t)
        d.text((a.width - 16, a.height + 7), "ㄱ자 일체형 진자 + M6 볼트 추 · 정지 상태에서 출발 · 실시간",
               fill=INK3, font=f_xs, anchor="rt")
        d.text((14, 38), "t = %.2f s" % t[k], fill=INK2, font=f_m)
        d.text((14, 58), "진자 %3.0f°" % ang[k], fill=C_ANG, font=f_m)
        d.text((14, 78), "토크 %+.2f N·m" % taus[k], fill=C_TAU, font=f_m)
        if t_up is not None and t[k] >= t_up:
            d.text((a.width - 14, 12), "도립 도달 %.2f s" % t_up, fill=C_UP, font=f_t, anchor="ra")
        frames.append(canvas)
        durs.append(int(round(a.every * dt * 1000)))
    r.close()
    durs[0] = 700
    durs[-1] = 1500

    if a.sheet:
        pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 8)]
        w, h = pick[0].size
        s = 0.5
        sh = Image.new("RGB", (int(w * s) * 4, int(h * s) * 2), BG)
        for i, im in enumerate(pick):
            sh.paste(im.resize((int(w * s), int(h * s))), ((i % 4) * int(w * s), (i // 4) * int(h * s)))
        sh.save(a.sheet)

    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True,
              disposal=1)
    stats.update(frames=len(frames), fps=1000.0 / durs[1], seconds=float(t[-1]),
                 mb=os.path.getsize(a.out) / 1e6, controller=name,
                 policy=None if a.classical else os.path.abspath(a.policy), al0=a.al0,
                 delay_ms=a.delay_ms, hist=hist)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, ensure_ascii=False)
    print("wrote %s  (%d frames, %.1f MB)" % (a.out, len(frames), stats["mb"]))


if __name__ == "__main__":
    main()
