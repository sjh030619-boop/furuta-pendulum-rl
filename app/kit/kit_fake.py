"""가짜 키트: CAN 버스와 AS5600 을 계산으로 흉내 냅니다 (kit_server.py --fake).

can0 과 I2C 에 아무것도 보내지 않습니다. 조원의 sim/fake_rig_run_policy.py 와 같은 방식으로
run_policy.Motor / Pendulum 의 버스를 바꿔 끼우고, canguard·session 점검도 가짜로 바꿉니다.

플랜트는 앱의 physics.js(= s2r/s2r_env.py 의 공칭 조건)를 그대로 옮긴 것이고 벽시계에 맞춰 진행합니다.
받침 기울기(--fake-tilt-deg, --fake-tilt-dir)를 주면 진자가 매달리는 방향이 팔 각도에 따라 바뀝니다:
중력 항이 sin(α − δ(θ)), δ(θ) = 기울기·cos(θ − 방향). 센서 영점(진자 자석 자리)은 session.ZERO_C + --fake-zero-shift.
상수는 kit_fake_model.json(없으면 ../src/assets/model.json)에서 읽습니다. numpy, mujoco 가 필요 없습니다.
"""
import os
import json
import math
import time
import types
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
RTT = 0.00042
THETA0_DEG = 61.29


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_model():
    for p in (os.path.join(HERE, "kit_fake_model.json"), os.path.join(HERE, "..", "src", "assets", "model.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError("kit_fake_model.json 이 없습니다 (app/src/assets/model.json 을 복사하세요)")


class Plant:
    """physics.js 의 substep 을 옮긴 것. locked 면 팔이 브레이크로 고정됩니다."""

    def __init__(self, C, al0=0.0, ald0=0.0, tilt_deg=0.0, tilt_dir=0.0):
        self.C = C
        self.tilt = math.radians(tilt_deg)
        self.tilt_dir = math.radians(tilt_dir)
        n = C["nominal"]
        self.p = n
        self.m00 = [C["m00"][i] + n["k"] * C["m00_k"][i] for i in range(5)]
        self.m01 = [C["m01"][i] + n["k"] * C["m01_k"][i] for i in range(3)]
        self.m00[0] += n["J"]
        self.fl = C["pend_fl"] * n["pend_fl"]
        self.dt = C["timestep"]
        self.th, self.al, self.thd, self.ald = 0.0, 0.0, 0.0, ald0
        self.al = self.delta() + al0
        self.tau = 0.0
        self.locked = True
        self.t = time.time()

    def delta(self):
        return self.tilt * math.cos(math.radians(THETA0_DEG) + self.th - self.tilt_dir)

    def substep(self):
        C, p = self.C, self.p
        al, thd, ald, dt = self.al, self.thd, self.ald, self.dt
        g = C["g1"]
        f1 = -C["pend_c2"] * ald * abs(ald) - C["pend_damping"] * ald - self.fl * math.tanh(ald / C["pend_fl_eps"])
        ca, sa = math.cos(al), math.sin(al)
        c2, s2 = 2 * ca * ca - 1, 2 * sa * ca
        ag = al - self.delta() if self.tilt else al
        c0, c1 = self.m00, self.m01
        m00 = c0[0] + c0[1] * ca + c0[2] * sa + c0[3] * c2 + c0[4] * s2
        m01 = c1[0] + c1[1] * ca + c1[2] * sa
        m11 = C["m11"]
        d00 = -c0[1] * sa + c0[2] * ca - 2 * c0[3] * s2 + 2 * c0[4] * c2
        d01 = -c1[1] * sa + c1[2] * ca
        grav = g[0] + g[1] * math.cos(ag) + g[2] * math.sin(ag)
        if self.locked:
            self.thd = 0.0
            self.ald = ald + dt * (f1 - grav) / m11
        else:
            fc = p["Fp"] if thd > 0 else p["Fm"]
            f0 = self.tau * p["kt"] - (fc * math.tanh(thd / C["arm_fric_eps"]) + p["b"] * thd)
            b0 = f0 - (d00 * thd * ald + d01 * ald * ald)
            b1 = f1 - (-0.5 * d00 * thd * thd + grav)
            det = m00 * m11 - m01 * m01
            self.thd = thd + dt * (b0 * m11 - b1 * m01) / det
            self.ald = ald + dt * (b1 * m00 - b0 * m01) / det
        self.th += dt * self.thd
        self.al += dt * self.ald

    def advance(self):
        now = time.time()
        if now - self.t > 1.0:
            self.t = now - 1.0
        while self.t + self.dt <= now:
            self.substep()
            self.t += self.dt


class FakeCAN:
    def __init__(self, plant, kt, temp=40):
        self.plant = plant
        self.kt = kt
        self.temp = temp
        self.q = collections.deque()
        self.brake = 0
        self.t_release = None
        self.sent = collections.Counter()
        self.mute = False
        self.prev_rep = 0

    def send(self, msg):
        P = self.plant
        P.advance()
        d = bytes(msg.data)
        c = d[0]
        self.sent[c] += 1
        if self.mute:
            return
        r = bytearray(8)
        r[0] = c
        a92 = THETA0_DEG + math.degrees(P.th)
        spd = int(round(math.degrees(P.thd)))
        if c == 0xA1:
            iq = int.from_bytes(d[4:6], "little", signed=True)
            rep = int(round(iq * P.p["kt"]))
            P.tau = 0.0 if P.locked else rep * 0.01 * self.kt / P.p["kt"]
            r[1] = self.temp
            r[2:4] = int(self.prev_rep).to_bytes(2, "little", signed=True)
            self.prev_rep = rep
            r[4:6] = spd.to_bytes(2, "little", signed=True)
            r[6:8] = (int(round(a92)) & 0xFFFF).to_bytes(2, "little")
        elif c == 0x92:
            r[4:8] = int(round(a92 * 100)).to_bytes(4, "little", signed=True)
        elif c == 0x9A:
            if self.t_release and time.time() - self.t_release > 0.2:
                self.brake = 1
                P.locked = False
            r[1], r[2], r[3] = self.temp, 52, self.brake
            r[4:6] = (476).to_bytes(2, "little")
        elif c == 0x9C:
            r[1] = self.temp
            r[4:6] = spd.to_bytes(2, "little", signed=True)
        elif c == 0x77:
            self.t_release = time.time()
        elif c == 0x78:
            self.brake = 0
            self.t_release = None
            P.locked = True
            P.tau = 0.0
            P.ald += 0.02
        elif c in (0x81, 0x80):
            P.tau = 0.0
        self.q.append((time.time() + RTT, types.SimpleNamespace(arbitration_id=0x241, data=r)))

    def recv(self, timeout=0.0):
        if self.q:
            ready, m = self.q[0]
            if ready - time.time() <= timeout:
                while time.time() < ready:
                    pass
                self.q.popleft()
                return m
        if timeout > 0:
            time.sleep(timeout)
        return None

    def shutdown(self):
        pass


class FakeI2C:
    def __init__(self, plant, zero_c):
        self.plant = plant
        self.zero_c = zero_c

    def read_i2c_block_data(self, addr, reg, n):
        self.plant.advance()
        raw = int(round(self.zero_c + wrap(self.plant.al) * 4096 / (2 * math.pi))) % 4096
        return [raw >> 8, raw & 0xFF]

    def close(self):
        pass


def install(rp, session, canguard, swing_deg=0.0, temp=40, tilt_deg=0.0, tilt_dir=0.0, zero_shift=0.0):
    """run_policy 모듈의 버스와 점검 함수를 가짜로 바꿉니다. 플랜트와 가짜 버스를 돌려줍니다."""
    import sys
    if "can" not in sys.modules:
        try:
            import can  # noqa: F401
        except ImportError:
            sys.modules["can"] = types.SimpleNamespace(
                Message=lambda arbitration_id, data, is_extended_id: types.SimpleNamespace(
                    arbitration_id=arbitration_id, data=data))
    C = load_model()
    plant = Plant(C, al0=math.radians(swing_deg), tilt_deg=tilt_deg, tilt_dir=tilt_dir)
    bus = FakeCAN(plant, rp.CONFIG["KT"], temp)
    i2c = FakeI2C(plant, session.ZERO_C + zero_shift)

    def motor_init(self, cfg):
        self.bus, self.cfg, self.req, self.rsp = bus, cfg, 0x141, 0x241

    def pend_init(self, cfg):
        self.bus, self.cfg, self.prev_a, self.prev_t = i2c, cfg, None, None

    def preflight(b, i=None, label=""):
        print("[가짜] 세션 점검 생략", label)
        a92 = THETA0_DEG + math.degrees(plant.th)
        out = {"label": label, "can0": "fake", "brake": bus.brake, "a92_start": a92}
        if i is not None:
            hi, lo = i.read_i2c_block_data(0x36, 0x0C, 2)
            out["pend_raw"] = ((hi & 0x0F) << 8) | lo
            out["pend_spread"] = 0
            out["pend_zero_expected"] = session.zero_raw(a92)
        return out

    rp.Motor.__init__ = motor_init
    rp.Pendulum.__init__ = pend_init
    canguard.fast_check = lambda *a, **k: None
    canguard.foreign_frame = lambda *a, **k: None
    session.preflight = preflight
    session.record_end_angle = lambda b: round(THETA0_DEG + math.degrees(plant.th), 2)
    return plant, bus
