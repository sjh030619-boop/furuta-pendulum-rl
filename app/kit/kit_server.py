#!/usr/bin/env python3
"""Pendulum Studio 키트 대기 프로그램 (규약 v2, app/PROTOCOL.md).

노트북의 앱이 USB 로 보낸 정책(신경망 가중치)을 받아 두었다가, 실행 명령이 오면 Pi 에서 50 Hz 로 직접 돌리고
그동안의 상태를 앱으로 보내 줍니다. 제어 루프가 Pi 안에서 돌기 때문에 USB 지연이 제어 지연이 되지 않습니다.

  python3 kit_server.py --serial /dev/ttyGS0           # USB 장치 모드 (Pi 의 USB-C)
  python3 kit_server.py --serial /dev/serial0          # USB-시리얼 어댑터 -> GPIO 14/15
  python3 kit_server.py --tcp 127.0.0.1:47800          # 시험용 (ssh -L 로 노트북에서 접속)
  python3 kit_server.py --tcp 127.0.0.1:47800 --fake   # 모터·센서 없이 계산으로만 (can0, I2C 에 아무것도 안 보냄)

모터 구동부는 새로 쓰지 않았습니다. 같은 폴더(또는 --host-dir)의 run_policy.py / session.py / canguard.py 를 불러서
Motor, Pendulum, speed_torque_limit, brake_limit, torque_to_iq, graceful_stop 을 그대로 씁니다.
제어 순서와 값은 실기에서 검증한 명령과 같습니다:  run_policy.py --run --tau 2.52 --iq-max 120  (제동 제한 5 W / 20°)

지키는 것
- CAN 버스는 실행(R)·점검(C) 동안에만 엽니다. 대기 중에는 can0 에 아무것도 보내지 않습니다.
- 실행 전: session.preflight (can0 상태, 다른 송신자, 모터 응답) -> 진자가 아래로 매달려 멈출 때까지 대기 -> 브레이크 해제
  -> 수평 측정 (팔을 120° 씩 두 번 돌려 세 자리에서 진자 영점을 잼 -> 받침 기울기 -> 이번 실행의 영점 식).
- 연결 보정(G, 앱에서 확인 단추를 눌렀을 때만): 영점 네 자리, 팔 마찰·관성 펄스, 진자 자유진동 주기.
- 실행 중 멈춤 조건: 팔 1100 dps, 진자 25 rad/s (run_policy 와 같음), 진자가 1.5 s 에 두 바퀴 넘게 돎(계속 회전),
  모터 응답 없음 0.1 s, 앱의 살아 있음 신호(A) 2 s 끊김, 정지 명령(X), 모터 온도 65°C 이상(시작 전).
- 끝날 때는 언제나 graceful_stop: iq 0 -> 0x81 -> 0x78 -> (속도 확인 뒤) 0x80.
- 수평 측정·보정 중 멈춤 조건: 팔 600 dps, 모터 응답 3번 연속 없음, 앱 신호 끊김, 정지 명령, 진자가 12 s 안에 안 멈춤.
- 0xB3 은 쓰지 않습니다.
"""
import os
import sys
import csv
import math
import time
import zlib
import queue
import base64
import socket
import signal
import argparse
import threading
import collections
import json

HERE = os.path.dirname(os.path.abspath(__file__))

NAME = "furuta-rev11"
VERSION = "2.1.0"
HZ = 50
TAU = 2.52
IQ_MAX = 120
BRAKE_W = 5.0
BRAKE_ZONE = math.radians(20.0)
MAX_SECONDS = 30.0
ALIVE_TIMEOUT = 2.0
TEMP_MAX = 65
SPIN_WINDOW = 1.5
SPIN_TURNS = 2.0
STILL_COUNTS = 6
HANG_COUNTS = 60
STILL_MAX_S = 60.0
OBS_N = 6
MAX_VALUES = 200000
LEVEL_IQ = 60
LEVEL_PULSE_S = 0.25
LEVEL_RUN = (3, 120.0)
LEVEL_CAL = (4, 90.0)
LEVEL_STILL = 3
LEVEL_SETTLE_MIN = 1.0
LEVEL_SETTLE_MAX = 12.0
LEVEL_RESID_MAX = 4.0
TILT_MAX_DEG = 5.0
LEVEL_ARM_DPS = 600
CAL_IQ_MAX = 100
SWING_S = 10.0
J_BODY = 2.358e-4

OPENED = object()
CLOSED = object()


class Stop(Exception):
    """실행을 끝내는 사유. reason 은 규약의 R END 낱말입니다."""

    def __init__(self, reason, detail=""):
        Exception.__init__(self, reason)
        self.reason = reason
        self.detail = detail


class Link:
    """줄 단위 전송. 읽기·쓰기는 스레드가 맡고 제어 루프는 큐만 봅니다."""

    def __init__(self):
        self.inq = queue.Queue()
        self.outq = queue.Queue(maxsize=4000)
        self.dropped = 0

    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()

    def send(self, line):
        try:
            self.outq.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def _reader(self):
        while True:
            self._open()
            self.inq.put(OPENED)
            buf = b""
            while True:
                try:
                    chunk = self._recv()
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", "replace").strip()
                    if s:
                        self.inq.put(s)
                if len(buf) > 8192:
                    buf = b""
            self.inq.put(CLOSED)
            self._close()
            time.sleep(0.3)

    def _writer(self):
        while True:
            s = self.outq.get()
            try:
                self._send((s + "\n").encode("utf-8"))
            except (OSError, AttributeError):
                pass


class TcpLink(Link):
    def __init__(self, host, port):
        Link.__init__(self)
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.sock = None
        self.where = "tcp %s:%d" % (host, port)

    def _open(self):
        s, _ = self.srv.accept()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def _recv(self):
        return self.sock.recv(4096)

    def _send(self, data):
        if self.sock is not None:
            self.sock.sendall(data)

    def _close(self):
        try:
            self.sock.close()
        except (OSError, AttributeError):
            pass
        self.sock = None


class SerialLink(Link):
    """직렬 포트 115200 8N1 raw. pyserial 없이 termios 로 엽니다(Pi 에서만)."""

    def __init__(self, path, baud=115200):
        Link.__init__(self)
        self.path = path
        self.baud = baud
        self.fd = None
        self.where = "serial %s %d" % (path, baud)

    def _open(self):
        import termios
        while True:
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY)
                a = termios.tcgetattr(fd)
                a[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP | termios.INLCR |
                          termios.IGNCR | termios.ICRNL | termios.IXON | termios.IXOFF)
                a[1] &= ~termios.OPOST
                a[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
                a[2] |= termios.CS8 | termios.CLOCAL | termios.CREAD
                a[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
                sp = getattr(termios, "B%d" % self.baud)
                a[4] = sp
                a[5] = sp
                a[6][termios.VMIN] = 1
                a[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, a)
                termios.tcflush(fd, termios.TCIOFLUSH)
                self.fd = fd
                return
            except OSError:
                time.sleep(1.0)

    def _recv(self):
        return os.read(self.fd, 1024)

    def _send(self, data):
        if self.fd is not None:
            os.write(self.fd, data)

    def _close(self):
        try:
            os.close(self.fd)
        except (OSError, TypeError):
            pass
        self.fd = None


class Actor:
    """앱의 MLP 와 같은 계산. 값은 층마다 W(행 우선, 행 = 출력) 다음 b 순서의 float32 입니다."""

    def __init__(self, np, sizes, act, squash, vals):
        self.np = np
        self.sizes = sizes
        self.act = act
        self.squash = squash
        self.layers = []
        k = 0
        for i in range(len(sizes) - 1):
            r, c = sizes[i + 1], sizes[i]
            W = vals[k:k + r * c].reshape(r, c)
            k += r * c
            b = vals[k:k + r]
            k += r
            self.layers.append((W, b))

    def __call__(self, obs):
        np = self.np
        x = np.asarray(obs, dtype=np.float64)
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < last:
                x = np.tanh(x) if self.act == "tanh" else np.maximum(0.0, x)
        u = float(x[0])
        if self.squash:
            u = math.tanh(u)
        return max(-1.0, min(1.0, u))


def torque_path(rp, cfg, u, theta_dot, alpha, tau_max=TAU, plim=BRAKE_W, zone=BRAKE_ZONE):
    """정책 출력 u -> iq. run_policy.py 본 루프와 같은 순서(보상 --comp 없음)."""
    tau_sim = rp.speed_torque_limit(u * tau_max, theta_dot, tau_max, cfg["w_no_load"])
    tau_real = tau_sim
    tau_unlim = tau_real
    braked = 0
    if plim is not None:
        lim = cfg["IQ_MAX"] * 0.01 * cfg["KT"]
        tau_c = max(-lim, min(lim, tau_real))
        tau_real = rp.brake_limit(tau_c, theta_dot, rp.wrap(alpha - math.pi), plim, zone)
        braked = int(abs(tau_real - tau_c) > 1e-9)
    iq = rp.torque_to_iq(tau_real, cfg)
    return iq, tau_real, tau_unlim, braked


class SpinWatch:
    """진자가 한쪽으로 계속 도는지 봅니다: 최근 window 초 동안 순회전이 turns 바퀴를 넘으면 True.
    2026-11-24 실기 기록: 정상 스윙업은 어느 구간에서도 0.96 바퀴 이하, 계속 돌 때는 1.5 s 에 3.2 바퀴."""

    def __init__(self, window=SPIN_WINDOW, turns=SPIN_TURNS):
        self.window = window
        self.limit = turns * 2 * math.pi
        self.hist = collections.deque()
        self.un = None
        self.prev = None

    def update(self, t, alpha):
        if self.prev is None:
            self.un = alpha
        else:
            self.un += (alpha - self.prev + math.pi) % (2 * math.pi) - math.pi
        self.prev = alpha
        self.hist.append((t, self.un))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        return abs(self.un - self.hist[0][1]) > self.limit


def load_host(host_dir):
    sys.path.insert(0, host_dir)
    import run_policy as rp
    import session
    import canguard
    return rp, session, canguard


class Server:
    def __init__(self, link, host, fake, log_dir):
        import numpy as np
        self.np = np
        self.link = link
        self.rp, self.session, self.canguard = host
        self.fake = fake
        self.log_dir = log_dir
        self.actor = None
        self.crc = None
        self.raw = None
        self.up = None
        self.last_end = "-"
        self.last_alive = 0.0
        self.alive_gap = 0.0
        self.miss = 0
        self.level_fit = None
        self.cal = self.load_cal()

    def send(self, line):
        self.link.send(line)

    def msg(self, code, text=""):
        self.send(("M %s %s" % (code, text)).rstrip())

    def info(self):
        self.send("I %s %s %d %d v2%s" % (NAME, VERSION, HZ, int(round(TAU * 1000)), " fake" if self.fake else ""))

    def status(self, state):
        self.send("K %s %s %s" % (state, self.crc or "-", self.last_end))

    def loop(self):
        print("대기 중:", self.link.where, "(가짜 장치)" if self.fake else "", flush=True)
        while True:
            line = self.link.inq.get()
            if line is OPENED or line is CLOSED:
                self.up = None
                continue
            try:
                self.handle(line)
            except Exception as e:
                self.send("E error %s" % str(e).replace("\n", " ")[:200])

    def handle(self, line):
        p = line.split()
        c = p[0]
        if c == "H":
            self.info()
        elif c == "Q":
            self.status("idle")
        elif c == "P":
            self.policy(p, line)
        elif c == "C":
            self.check()
        elif c == "L":
            self.send(self.cal_line("L"))
        elif c == "G":
            self.calibrate()
        elif c == "R":
            try:
                secs = float(p[1]) if len(p) > 1 else 10.0
            except ValueError:
                secs = -1.0
            if not 1.0 <= secs <= MAX_SECONDS:
                self.send("E run seconds 1..%d" % MAX_SECONDS)
                return
            self.run(secs, " ".join(p[2:]))
        elif c in ("X", "A"):
            pass
        else:
            self.send("E cmd %s" % c[:20])

    def policy(self, p, line):
        np = self.np
        if len(p) >= 2 and p[1] == "BEGIN":
            try:
                n = int(p[2])
                crc = p[3].lower()
                act = p[4]
                squash = p[5] == "1"
                sizes = [int(x) for x in p[6].split(",")]
            except (IndexError, ValueError):
                self.send("E policy header")
                return
            need = sum(sizes[i + 1] * sizes[i] + sizes[i + 1] for i in range(len(sizes) - 1))
            if (act not in ("relu", "tanh") or len(sizes) < 2 or len(sizes) > 5 or sizes[0] != OBS_N or sizes[-1] != 1
                    or max(sizes) > 256 or n != need or n > MAX_VALUES):
                self.send("E policy shape")
                return
            self.up = dict(n=n, crc=crc, act=act, squash=squash, sizes=sizes, parts=[])
            return
        if self.up is None:
            self.send("E policy nobegin")
            return
        if len(p) >= 2 and p[1] == "END":
            up, self.up = self.up, None
            try:
                raw = base64.b64decode("".join(up["parts"]), validate=True)
            except (ValueError, TypeError):
                self.send("E policy base64")
                return
            if len(raw) != 4 * up["n"]:
                self.send("E policy length %d" % (len(raw) // 4))
                return
            crc = "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
            if crc != up["crc"]:
                self.send("E policy crc %s" % crc)
                return
            vals = np.frombuffer(raw, dtype="<f4").astype(np.float64)
            if not np.all(np.isfinite(vals)) or float(np.max(np.abs(vals))) > 1e4:
                self.send("E policy values")
                return
            self.actor = Actor(np, up["sizes"], up["act"], up["squash"], vals)
            self.crc = crc
            self.raw = raw
            self.send("P OK %s" % crc)
            return
        if len(p) == 2:
            self.up["parts"].append(p[1])
            if sum(len(x) for x in self.up["parts"]) > 6 * MAX_VALUES:
                self.up = None
                self.send("E policy toolong")
            return
        self.send("E policy line")

    def poll(self):
        """실행·대기 중에 들어온 줄을 봅니다. 정지(X)와 연결 끊김은 Stop 으로 올립니다."""
        while True:
            try:
                s = self.link.inq.get_nowait()
            except queue.Empty:
                return
            if s is OPENED:
                continue
            if s is CLOSED:
                raise Stop("link", "closed")
            if s == "A":
                now = time.time()
                self.alive_gap = max(self.alive_gap, now - self.last_alive)
                self.last_alive = now
            elif s == "X":
                raise Stop("stop")
            elif s == "Q":
                self.status("running")
            elif s == "H":
                self.info()
            else:
                self.send("E busy %s" % s.split()[0][:20])

    def alive_check(self):
        self.poll()
        if time.time() - self.last_alive > ALIVE_TIMEOUT:
            raise Stop("link", "no alive %.1f s" % ALIVE_TIMEOUT)

    def wait_hanging(self, pend, a92, C=None):
        rp = self.rp
        z = self.session.zero_raw(a92)
        if C is not None:
            z += C - self.session.ZERO_C
        t0 = time.time()
        while time.time() - t0 < STILL_MAX_S:
            v = []
            t1 = time.time()
            while time.time() - t1 < 0.5:
                v.append(rp.pend_raw_now(pend))
                time.sleep(0.005)
            self.alive_check()
            spread = max(v) - min(v)
            if spread <= STILL_COUNTS and abs(v[-1] - z) <= HANG_COUNTS:
                return time.time() - t0
            self.msg("still", "%d %d" % (int(time.time() - t0), spread))
        raise Stop("not_still", "raw %d zero %.0f" % (v[-1], z))

    def load_cal(self):
        try:
            with open(os.path.join(self.log_dir, "kit_cal_latest.json"), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def zero_c(self):
        """진자 자석 영점 C: 연결 보정 값이 있으면 그것, 없으면 session.py 의 값."""
        if self.cal and self.cal.get("level"):
            return float(self.cal["level"]["C"])
        return float(self.session.ZERO_C)

    def cal_line(self, head):
        k = self.cal
        if not k:
            return "%s NONE C=%.1f" % (head, self.zero_c())
        a, lv, sw = k.get("arm") or {}, k.get("level") or {}, k.get("swing") or {}
        return ("%s %s C=%.1f tilt=%.2f dir=%.0f J=%.4f Jm=%.4f Fp=%.3f Fm=%.3f b=%.3f T0=%s T0m=%.4f temp=%d ok=%d" % (
            head, k["stamp"], lv.get("C", self.zero_c()), lv.get("tilt_deg", 0.0), lv.get("phi_deg", 0.0),
            a.get("J", 0.0), a.get("J_motor", 0.0), a.get("Fc_pos", 0.0), a.get("Fc_neg", 0.0), a.get("viscous", 0.0),
            ("%.4f" % sw["T0"]) if sw.get("ok") else "-", k.get("T0_model", 0.0), k.get("temp_C", -1),
            1 if k.get("valid") else 0))

    def tick(self, motor, pend, iq, nxt):
        """수평 측정·보정의 10 ms 주기 하나: 토크 명령을 보내고 진자 RAW 를 읽습니다."""
        now = time.time()
        if now < nxt:
            time.sleep(nxt - now)
        self.alive_check()
        st = motor.send_torque(iq)
        if st is None:
            self.miss += 1
            if self.miss >= 3:
                raise Stop("watchdog", "motor silent")
        else:
            self.miss = 0
            if abs(st["speed_dps"]) > LEVEL_ARM_DPS:
                raise Stop("arm_speed", "%d dps" % st["speed_dps"])
        return st, self.rp.pend_raw_now(pend), max(nxt + 0.01, time.time() - 0.005)

    def settle(self, motor, pend, nxt, min_s=LEVEL_SETTLE_MIN):
        """iq 0 으로 팔과 진자가 멈출 때까지: 0.5 s 동안 진자 RAW 변동 <= LEVEL_STILL, 팔 속도 0."""
        t0 = time.time()
        win = collections.deque()
        while True:
            st, raw, nxt = self.tick(motor, pend, 0, nxt)
            now = time.time()
            win.append((now, raw, abs(st["speed_dps"]) if st else 99))
            while win and now - win[0][0] > 0.5:
                win.popleft()
            if now - t0 >= max(min_s, 0.5):
                r = [x[1] for x in win]
                if max(r) - min(r) <= LEVEL_STILL and max(x[2] for x in win) == 0:
                    return float(sorted(r)[len(r) // 2]), now - t0, nxt
            if now - t0 > LEVEL_SETTLE_MAX:
                raise Stop("not_still", "level stop %.0f s" % (now - t0))

    def move(self, motor, pend, deg, nxt):
        """팔을 +방향으로 deg 만큼: iq LEVEL_IQ 펄스 LEVEL_PULSE_S 를 반복 (조원 exp2_arm_steps 의 훑기와 같은 펄스)."""
        a0 = motor.read_angle92()
        if a0 is None:
            raise Stop("motor", "no 0x92 reply")
        for _ in range(10):
            t1 = time.time()
            while time.time() - t1 < LEVEL_PULSE_S:
                _, _, nxt = self.tick(motor, pend, LEVEL_IQ, nxt)
            t2 = time.time()
            while time.time() - t2 < 1.5:
                st, _, nxt = self.tick(motor, pend, 0, nxt)
                if st and st["speed_dps"] == 0 and time.time() - t2 > 0.1:
                    break
            a = motor.read_angle92()
            if a is not None and a - a0 >= deg - 12.0:
                return nxt
        raise Stop("motor", "arm did not move %.0f deg" % deg)

    def level(self, motor, pend, n, step, C, head):
        """팔을 step 씩 돌려 n 자리에서 진자 영점 RAW 를 잽니다. C 를 주면 A, φ 만, 안 주면 C 도 맞춤."""
        import kit_cal
        stops = []
        nxt = time.time()
        t0 = time.time()
        for i in range(n):
            if i > 0:
                nxt = self.move(motor, pend, step, nxt)
            raw, waited, nxt = self.settle(motor, pend, nxt, LEVEL_SETTLE_MIN if i > 0 else 0.5)
            a92 = motor.read_angle92()
            if a92 is None:
                raise Stop("motor", "no 0x92 reply")
            stops.append((a92, raw))
            self.send("%s level %d/%d %.2f %.0f %.1f" % (head, i + 1, n, a92, raw, waited))
        fit = kit_cal.fit_level(stops, C)
        fit["seconds"] = round(time.time() - t0, 1)
        fit["stops"] = stops
        return fit

    def read_alpha(self, pend, t, a92, zero):
        """run_policy.Pendulum.read_alpha 와 같은 계산. 영점만 이번 실행의 수평 측정 식(zero)을 씁니다."""
        rp = self.rp
        raw = rp.pend_raw_now(pend)
        pend.last_raw = raw
        pend.last_zero = zero(a92)
        sgn = pend.cfg["alpha_sign"] if pend.cfg["alpha_sign"] is not None else +1
        a = rp.wrap(sgn * (raw - pend.last_zero) * 2 * math.pi / 4096.0)
        ad = 0.0
        if pend.prev_a is not None and t > pend.prev_t:
            ad = rp.wrap(a - pend.prev_a) / (t - pend.prev_t)
        pend.prev_a, pend.prev_t = a, t
        return a, ad

    def calibrate(self):
        """연결 보정 (앱에서 확인 단추를 눌렀을 때만): 영점 네 자리 -> 팔 마찰·관성 펄스 -> 팔을 잠그고 진자 자유진동."""
        import kit_cal
        rp, session, canguard = self.rp, self.session, self.canguard
        cfg = rp.CONFIG
        cfg["IQ_MAX"] = IQ_MAX
        motor = pend = None
        moved = False
        out = dict(stamp=time.strftime("%Y%m%d_%H%M%S"), made_by="kit_server.py G", fake=self.fake)
        frows, swing = [], []
        why = None
        self.last_alive = time.time()
        t0 = time.time()
        try:
            self.msg("check")
            pend = rp.Pendulum(cfg)
            motor = rp.Motor(cfg)
            pre = session.preflight(motor.bus, pend.bus, "kit_server G") or {}
            self.msg("still")
            self.wait_hanging(pend, pre.get("a92_start"), self.zero_c())
            self.msg("brake")
            moved = True
            motor.send_brake_release()
            motor.wait_brake_released()
            st = motor.send_torque(0)
            if st is None:
                raise Stop("motor", "no reply")
            if st["temp"] >= TEMP_MAX:
                raise Stop("hot", "%d C" % st["temp"])
            out["temp_C"] = st["temp"]
            lv = self.level(motor, pend, LEVEL_CAL[0], LEVEL_CAL[1], None, "G")
            out["level"] = lv
            self.send("G level ok C=%.1f tilt=%.2f dir=%.0f resid=%.1f" % (lv["C"], lv["tilt_deg"], lv["phi_deg"], lv["resid"]))
            self.send("G arm")
            rp.measure_friction.last = None
            try:
                rp.measure_friction(motor, pend, cfg, CAL_IQ_MAX, frows)
            except RuntimeError as e:
                if not frows or "응답 없음" in str(e) or "한계" in str(e):
                    raise Stop("error", "friction: %s" % e)
                why = str(e)
            self.alive_check()
            arm = kit_cal.fit_arm([r[:4] for r in frows], cfg["KT"])
            arm["J_motor"] = arm["J"] - J_BODY
            arm["bridge"] = {k: v for k, v in (getattr(rp.measure_friction, "last", None) or {}).items() if k != "points"}
            out["arm"] = arm
            self.send("G arm ok J=%.4f Fp=%.3f Fm=%.3f b=%.3f rms=%.1f" % (
                arm["J"], arm["Fc_pos"], arm["Fc_neg"], arm["viscous"], arm["speed_rms_dps"]))
            rp.graceful_stop(motor, cfg)
            moved = False
            self.send("G swing")
            ts = time.time()
            while time.time() - ts < SWING_S:
                self.alive_check()
                swing.append((time.time() - ts, rp.pend_raw_now(pend)))
                time.sleep(0.005)
            sw = kit_cal.fit_swing([x[0] for x in swing], [x[1] for x in swing])
            out["swing"] = sw
            try:
                import kit_fake
                out["T0_model"] = kit_cal.model_period(kit_fake.load_model())
            except Exception:
                out["T0_model"] = 0.0
            self.send("G swing ok T0=%s n=%d amp=%s" % (("%.4f" % sw["T0"]) if sw.get("ok") else "-", sw.get("n", 0),
                                                       ("%.0f" % sw["amp_max"]) if sw.get("ok") else "-"))
            bad = []
            if why:
                bad.append(why)
            if lv["resid"] > LEVEL_RESID_MAX:
                bad.append("level resid %.1f" % lv["resid"])
            if not sw.get("ok"):
                bad.append("swing too small")
            out["invalid_reasons"] = bad
            out["valid"] = not bad
            out["seconds"] = round(time.time() - t0, 1)
            self.cal = out
            self.send(self.cal_line("G OK"))
        except Stop as e:
            self.send(("G FAIL %s %s" % (e.reason, e.detail)).rstrip())
        except (session.PreflightError, canguard.CanGuardError) as e:
            self.send("G FAIL check %s" % str(e).replace("\n", " ")[:300])
        except Exception as e:
            self.send("G FAIL error %s: %s" % (type(e).__name__, str(e).replace("\n", " ")[:300]))
        finally:
            if motor is not None and moved:
                try:
                    rp.graceful_stop(motor, cfg)
                except Exception as e:
                    print("graceful_stop 실패:", e)
            if motor is not None:
                try:
                    print("마지막 0x92 각도 저장:", session.record_end_angle(motor.bus))
                except Exception as e:
                    print("마지막 각도 저장 실패:", e)
            self.close(motor, pend)
        self.save_cal(out, frows, swing)

    def save_cal(self, out, frows, swing):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            base = os.path.join(self.log_dir, "kit_cal_%s" % out["stamp"])
            if frows:
                with open(base + "_arm.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["t", "iq_cmd", "iq_reply", "speed_dps", "temp_C", "pend_raw"])
                    w.writerows(frows)
            if swing:
                with open(base + "_swing.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["t", "pend_raw"])
                    w.writerows(swing)
            with open(base + ".json", "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=1)
            if out.get("valid"):
                with open(os.path.join(self.log_dir, "kit_cal_latest.json"), "w", encoding="utf-8") as fh:
                    json.dump(out, fh, ensure_ascii=False, indent=1)
            print("보정 기록:", base + ".json", "valid" if out.get("valid") else out.get("invalid_reasons"), flush=True)
        except Exception as e:
            print("보정 기록 저장 실패:", e, flush=True)

    def check(self):
        """점검: 모터를 움직이지 않습니다 (0x9A, 0x92, 0x9C 읽기와 AS5600 읽기만)."""
        rp, session, canguard = self.rp, self.session, self.canguard
        motor = pend = None
        try:
            pend = rp.Pendulum(rp.CONFIG)
            motor = rp.Motor(rp.CONFIG)
            out = session.preflight(motor.bus, pend.bus, "kit_server C")
            d = motor.xfer(bytes([0x9A, 0, 0, 0, 0, 0, 0, 0]))
            temp = d[1] if d is not None else -1
            raw = out.get("pend_raw", -1)
            zero = out.get("pend_zero_expected", session.zero_raw(None))
            self.send("C OK temp=%d raw=%d zero=%.0f spread=%d a92=%.1f brake=%s" % (
                temp, raw, zero, out.get("pend_spread", -1), out.get("a92_start", 0.0), out.get("brake", "-")))
        except (session.PreflightError, canguard.CanGuardError) as e:
            self.send("C FAIL check %s" % str(e).replace("\n", " ")[:300])
        except Exception as e:
            self.send("C FAIL error %s: %s" % (type(e).__name__, str(e).replace("\n", " ")[:300]))
        finally:
            self.close(motor, pend)

    def close(self, motor, pend):
        if motor is not None:
            try:
                motor.shutdown()
            except Exception as e:
                print("버스 닫기 실패:", e)
        if pend is not None:
            try:
                pend.bus.close()
            except Exception as e:
                print("I2C 닫기 실패:", e)

    def run(self, seconds, note):
        rp, session, canguard = self.rp, self.session, self.canguard
        if self.actor is None:
            self.last_end = "nopolicy"
            self.send("R END nopolicy 0")
            return
        cfg = rp.CONFIG
        cfg["IQ_MAX"] = IQ_MAX
        actor = self.actor
        crc = self.crc
        rows = []
        reason, detail, steps = "error", "", 0
        self.level_fit = None
        motor = pend = None
        moved = False
        late = 0
        self.last_alive = time.time()
        try:
            self.msg("check")
            pend = rp.Pendulum(cfg)
            motor = rp.Motor(cfg)
            pre = session.preflight(motor.bus, pend.bus, "kit_server R") or {}
            self.msg("still")
            C = self.zero_c()
            self.wait_hanging(pend, pre.get("a92_start"), C)
            self.msg("brake")
            moved = True
            motor.send_brake_release()
            motor.wait_brake_released()
            self.alive_check()
            state = motor.send_torque(0)
            if state is None:
                raise Stop("motor", "no reply")
            if state["temp"] >= TEMP_MAX:
                raise Stop("hot", "%d C" % state["temp"])
            self.msg("level")
            self.level_fit = lv = self.level(motor, pend, LEVEL_RUN[0], LEVEL_RUN[1], C, "M")
            self.msg("level", "ok tilt=%.2f dir=%.0f resid=%.1f sec=%.1f" % (lv["tilt_deg"], lv["phi_deg"], lv["resid"], lv["seconds"]))
            if lv["resid"] > LEVEL_RESID_MAX:
                raise Stop("level", "resid %.1f counts" % lv["resid"])
            if lv["tilt_deg"] > TILT_MAX_DEG:
                raise Stop("tilt", "%.2f deg dir %.0f" % (lv["tilt_deg"], lv["phi_deg"]))
            import kit_cal
            zero = kit_cal.zero_fn(C, lv["A"], lv["phi_deg"])
            state = motor.send_torque(0) or state
            theta_start = motor.read_angle92()
            if theta_start is None:
                raise Stop("motor", "no 0x92 reply")
            state["a92"] = theta_start
            pend.prev_a = None
            spin = SpinWatch()
            dt = cfg["dt"]
            self.send("R START %s" % crc)
            self.last_alive = time.time()
            self.alive_gap = 0.0
            t0 = time.time()
            next_t = t0
            last_ok = t0
            while time.time() - t0 < seconds:
                now = time.time()
                if now < next_t:
                    time.sleep(next_t - now)
                    now = time.time()
                elif now - next_t > 0.005:
                    late += 1
                next_t += dt
                self.poll()
                if now - self.last_alive > ALIVE_TIMEOUT:
                    raise Stop("link", "no alive %.1f s" % ALIVE_TIMEOUT)

                alpha, alpha_dot = self.read_alpha(pend, now, state.get("a92"), zero)
                theta = math.radians(state["a92"] - theta_start)
                theta_dot = math.radians(state["speed_dps"])
                if abs(alpha_dot) > cfg["pend_speed_limit"]:
                    raise Stop("pend_speed", "%.1f rad/s" % alpha_dot)
                if abs(state["speed_dps"]) > cfg["arm_speed_limit_dps"]:
                    raise Stop("arm_speed", "%d dps" % state["speed_dps"])
                if spin.update(now, alpha):
                    raise Stop("spin", "%.1f turns in %.1f s" % (SPIN_TURNS, SPIN_WINDOW))

                obs = [math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha),
                       theta_dot / 20.0, alpha_dot / 40.0]
                u = actor(obs)
                iq, tau_real, tau_unlim, braked = torque_path(rp, cfg, u, theta_dot, alpha)
                tau_q = iq * 0.01 * cfg["KT"]
                brake_w = abs(tau_q * theta_dot) if tau_q * theta_dot < 0 else 0.0
                rows.append([round(now - t0, 3), theta, theta_dot, alpha, alpha_dot, u, iq,
                             state["angle_deg"], state.get("a92", ""), state["speed_dps"], pend.last_raw,
                             round(pend.last_zero, 1), state["iq"], state["temp"], round(tau_real, 4),
                             round(tau_unlim, 4), round(brake_w, 3), braked, ""])

                new = motor.send_torque(iq)
                if new is None:
                    if time.time() - last_ok > cfg["watchdog"]:
                        raise Stop("watchdog", "motor silent")
                else:
                    a92 = motor.read_angle92()
                    new["a92"] = a92 if a92 is not None else state.get("a92")
                    state = new
                    last_ok = time.time()
                steps += 1
                self.send("T %d %d %d %d %d %d %d" % (
                    round((now - t0) * 1000), round(math.degrees(theta) * 1000), round(math.degrees(alpha) * 1000),
                    round(math.degrees(theta_dot) * 1000), round(math.degrees(alpha_dot) * 1000),
                    round(tau_q / TAU * 1000), 32 if braked else 0))
            reason = "done"
        except Stop as e:
            reason, detail = e.reason, e.detail
        except (session.PreflightError, canguard.CanGuardError) as e:
            reason, detail = "check", str(e)
        except Exception as e:
            reason, detail = "error", "%s: %s" % (type(e).__name__, e)
        finally:
            if motor is not None and moved:
                try:
                    rp.graceful_stop(motor, cfg)
                except Exception as e:
                    print("graceful_stop 실패:", e)
                    detail = (detail + " / stop failed: %s" % e).strip()
            if motor is not None:
                try:
                    print("마지막 0x92 각도 저장:", session.record_end_angle(motor.bus))
                except Exception as e:
                    print("마지막 각도 저장 실패:", e)
            self.close(motor, pend)
        detail = str(detail).replace("\n", " ")[:300]
        self.last_end = reason
        self.send(("R END %s %d %s" % (reason, steps, detail)).rstrip())
        print("실행 끝: %s, %d 스텝, 늦은 주기 %d, 보내지 못한 줄 %d, 앱 신호 최대 간격 %.2f s %s" % (
            reason, steps, late, self.link.dropped, self.alive_gap, detail), flush=True)
        self.save(rows, reason, detail, seconds, note, crc, late)

    def save(self, rows, reason, detail, seconds, note, crc, late):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            base = os.path.join(self.log_dir, "kit_server_%s" % time.strftime("%Y%m%d_%H%M%S"))
            with open(base + ".csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t", "theta", "theta_dot", "alpha", "alpha_dot", "u", "iq", "angle_deg", "a92_deg",
                            "speed_dps", "pend_raw", "pend_zero_raw", "iq_reply_prev", "temp_C", "tau_cmd",
                            "tau_unlim", "brake_w", "brake_limited", "st1_hex"])
                w.writerows(rows)
            lv = self.level_fit or {}
            with open(base + ".txt", "w", encoding="utf-8") as fh:
                fh.write("end %s\ndetail %s\nseconds %s\nnote %s\npolicy_crc %s\nlate_cycles %d\nfake %s\n" % (
                    reason, detail, seconds, note, crc, late, self.fake))
                fh.write("level C %s A %s phi_deg %s tilt_deg %s resid %s seconds %s stops %s\n" % (
                    lv.get("C"), lv.get("A"), lv.get("phi_deg"), lv.get("tilt_deg"), lv.get("resid"), lv.get("seconds"),
                    lv.get("stops")))
            if self.raw is not None and self.actor is not None and crc == self.crc:
                a = self.actor
                if len(a.layers) == 3 and a.act == "relu" and a.squash:
                    f = self.np.float32
                    self.np.savez(base + "_policy.npz", **{
                        "latent_pi.0.weight": a.layers[0][0].astype(f), "latent_pi.0.bias": a.layers[0][1].astype(f),
                        "latent_pi.2.weight": a.layers[1][0].astype(f), "latent_pi.2.bias": a.layers[1][1].astype(f),
                        "mu.weight": a.layers[2][0].astype(f), "mu.bias": a.layers[2][1].astype(f)})
            print("기록:", base + ".csv", flush=True)
        except Exception as e:
            print("기록 저장 실패:", e, flush=True)


def single_instance():
    """대기 프로그램이 두 개 뜨지 않게 합니다 (CAN 명령은 한 프로세스에서만)."""
    try:
        import fcntl
    except ImportError:
        return None
    fh = open("/tmp/kit_server.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("kit_server.py 가 이미 실행 중입니다.")
        raise SystemExit(1)
    return fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="직렬 포트 (예: /dev/ttyGS0, /dev/serial0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--tcp", default=None, help="주소:포트 (시험용, 예: 127.0.0.1:47800)")
    ap.add_argument("--fake", action="store_true", help="모터·센서를 계산으로 흉내 (can0, I2C 사용 안 함)")
    ap.add_argument("--fake-swing-deg", type=float, default=0.0, help="--fake: 진자를 이만큼 흔들린 상태로 시작")
    ap.add_argument("--fake-temp", type=int, default=40, help="--fake: 모터 온도")
    ap.add_argument("--fake-tilt-deg", type=float, default=0.0, help="--fake: 받침 기울기")
    ap.add_argument("--fake-tilt-dir", type=float, default=0.0, help="--fake: 기울기 방향 (팔 0x92 각도 기준)")
    ap.add_argument("--fake-zero-shift", type=float, default=0.0, help="--fake: 진자 자석 영점이 session.py 와 다른 정도 (카운트)")
    ap.add_argument("--host-dir", default=None,
                    help="run_policy.py, session.py, canguard.py 가 있는 폴더 (기본: 이 폴더, 없으면 ../pi_host)")
    ap.add_argument("--log-dir", default=None, help="기본: <host-dir>/logs (가짜 장치는 <host-dir>/logs/fake)")
    a = ap.parse_args()
    if bool(a.serial) == bool(a.tcp):
        ap.error("--serial 과 --tcp 중 하나를 고르세요")
    if a.host_dir is None:
        a.host_dir = HERE if os.path.exists(os.path.join(HERE, "run_policy.py")) else os.path.join(HERE, "..", "pi_host")
    a.host_dir = os.path.abspath(a.host_dir)
    lock = single_instance()

    def on_term(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    host = load_host(a.host_dir)
    if a.fake:
        import kit_fake
        kit_fake.install(host[0], host[1], host[2], a.fake_swing_deg, a.fake_temp, a.fake_tilt_deg, a.fake_tilt_dir,
                         a.fake_zero_shift)
    log_dir = a.log_dir or os.path.join(a.host_dir, "logs", "fake" if a.fake else "")
    if a.tcp:
        h, _, port = a.tcp.rpartition(":")
        link = TcpLink(h or "127.0.0.1", int(port))
    else:
        link = SerialLink(a.serial, a.baud)
    link.start()
    try:
        Server(link, host, a.fake, log_dir).loop()
    except KeyboardInterrupt:
        print("끝냄")
    finally:
        del lock


if __name__ == "__main__":
    main()
