"""가짜 장치로 끝까지 시험: kit_server.py --fake 를 띄우고 앱이 하는 대로 줄을 주고받습니다.

  python test_fake.py <host-dir> <정책.npz> [--port 47811]

can0, I2C, 모터를 쓰지 않습니다. 확인하는 것: 인사, 정책 올리기(정상·CRC 틀림·모양 틀림), 실행해서 세움,
상태 줄 주기, 정지 명령, 살아 있음 신호 끊김, 정책 없이 실행, 흔들리는 진자 기다림, 모터 과열.
"""
import os
import sys
import time
import zlib
import base64
import socket
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def pack(npz):
    w = np.load(npz)
    parts = [w["latent_pi.0.weight"], w["latent_pi.0.bias"], w["latent_pi.2.weight"], w["latent_pi.2.bias"],
             w["mu.weight"], w["mu.bias"]]
    return np.concatenate([np.asarray(x, dtype="<f4").ravel() for x in parts]).astype("<f4").tobytes()


class Client:
    def __init__(self, port):
        for _ in range(50):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("서버에 접속 못 함")
        self.buf = b""

    def tx(self, line):
        self.s.sendall((line + "\n").encode("utf-8"))

    def rx(self, timeout=2.0):
        end = time.time() + timeout
        while b"\n" not in self.buf:
            left = end - time.time()
            if left <= 0:
                return None
            self.s.settimeout(left)
            try:
                d = self.s.recv(65536)
            except socket.timeout:
                return None
            if not d:
                return None
            self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("utf-8").strip()

    def ask(self, line, timeout=2.0):
        self.tx(line)
        return self.rx(timeout)

    def upload(self, raw, crc=None, sizes="6,64,64,1"):
        crc = crc or "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
        self.tx("P BEGIN %d %s relu 1 %s" % (len(raw) // 4, crc, sizes))
        b = base64.b64encode(raw).decode()
        for i in range(0, len(b), 180):
            self.tx("P " + b[i:i + 180])
        return self.ask("P END", 5.0)

    def run(self, secs, alive=True, stop_at=None, alive_until=None):
        """R 을 보내고 끝날 때까지 받습니다. (끝 줄, 상태 줄들, 안내 줄들, R START 까지 걸린 시간)"""
        self.tx("R %g test" % secs)
        t0 = time.time()
        T, M, start, last_a, stopped = [], [], None, 0.0, False
        while True:
            now = time.time()
            quiet = alive_until is not None and start is not None and now - start > alive_until
            if alive and not quiet and now - last_a > 0.5:
                self.tx("A")
                last_a = now
            if stop_at is not None and start is not None and not stopped and now - start > stop_at:
                self.tx("X")
                stopped = True
            line = self.rx(0.1)
            if line is None:
                if now - t0 > secs + 60:
                    return "TIMEOUT", T, M, start
                continue
            if line.startswith("T "):
                T.append([int(x) for x in line.split()[1:]])
            elif line.startswith("M "):
                M.append(line)
            elif line.startswith("R START"):
                start = time.time()
            elif line.startswith("R END"):
                return line, T, M, (start - t0) if start else None
            else:
                M.append(line)

    def cal(self, stop_at=None):
        """G 를 보내고 G OK / G FAIL 까지 받습니다. (끝 줄, 중간 줄들, 걸린 시간)"""
        self.tx("G")
        t0 = time.time()
        mid, last_a, stopped = [], 0.0, False
        while time.time() - t0 < 150:
            if time.time() - last_a > 0.5:
                self.tx("A")
                last_a = time.time()
            if stop_at is not None and not stopped and time.time() - t0 > stop_at:
                self.tx("X")
                stopped = True
            line = self.rx(0.1)
            if line is None:
                continue
            if line.startswith("G OK") or line.startswith("G FAIL"):
                return line, mid, time.time() - t0
            mid.append(line)
        return "TIMEOUT", mid, time.time() - t0

    def close(self):
        self.s.close()


def kv(line):
    out = {}
    for w in line.split():
        if "=" in w:
            k, v = w.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
    return out


LOGS = os.path.join(os.environ.get("TEMP", "/tmp"), "kit_fake_logs")


def server(host_dir, port, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "kit_server.py"), "--fake", "--tcp", "127.0.0.1:%d" % port,
           "--host-dir", host_dir, "--log-dir", LOGS] + list(extra)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def stood(T):
    """마지막 2초 동안 직립 ±10° 안인가"""
    tail = T[-100:]
    return all(abs((r[2] / 1000.0) % 360.0 - 180.0) < 10.0 for r in tail)


def finish(sv):
    sv.terminate()
    try:
        return sv.communicate(timeout=10)[0].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        sv.kill()
        return ""


def main():
    a = sys.argv[1:]
    port = 47811
    if "--port" in a:
        i = a.index("--port")
        port = int(a[i + 1])
        del a[i:i + 2]
    host_dir, npz = a[0], a[1]
    raw = pack(npz)
    crc = "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
    res = []

    def check(name, ok, note=""):
        res.append(bool(ok))
        print("%s  %s  %s" % ("통과" if ok else "실패", name, note), flush=True)

    try:
        os.remove(os.path.join(LOGS, "kit_cal_latest.json"))
    except OSError:
        pass
    sv = server(host_dir, port)
    out = ""
    try:
        c = Client(port)
        r = c.ask("H")
        check("인사", r is not None and r.startswith("I furuta-rev11") and " v2" in r, r)
        r = c.ask("Q")
        check("상태(정책 없음)", r == "K idle - -", r)
        r, _, _, _ = c.run(5)
        check("정책 없이 실행", r == "R END nopolicy 0", r)
        r = c.upload(raw, crc="deadbeef")
        check("CRC 틀린 정책 거절", r is not None and r.startswith("E policy crc"), r)
        r = c.upload(raw[:-4])
        check("길이 틀린 정책 거절", r == "E policy shape", r)
        c.rx(0.5)
        while c.rx(0.2) is not None:
            pass
        bad = np.frombuffer(raw, dtype="<f4").copy()
        bad[10] = np.nan
        r = c.upload(bad.tobytes())
        check("숫자가 아닌 값 거절", r == "E policy values", r)
        t0 = time.time()
        r = c.upload(raw)
        check("정책 올리기", r == "P OK " + crc, "%s (%.2f s, %d 바이트)" % (r, time.time() - t0, len(raw)))
        r = c.ask("Q")
        check("상태(정책 있음)", r == "K idle %s nopolicy" % crc, r)

        r, T, M, lead = c.run(10)
        ts = np.array([x[0] for x in T]) / 1000.0
        d = np.diff(ts) * 1000.0
        check("실행 10초", r is not None and r.startswith("R END done"), "%s, 상태 줄 %d개, 시작까지 %.1f s, 안내 %s" % (
            r, len(T), lead or -1, [m.split()[1] for m in M]))
        check("세움 (마지막 2초 직립 ±10°)", len(T) > 400 and stood(T))
        lv = [m for m in M if m.startswith("M level ok")]
        check("실행 전 수평 측정 (평평한 받침)", len(lv) == 1 and kv(lv[0]).get("tilt", 9) < 0.3, lv[0] if lv else M)
        check("상태 줄 주기 50 Hz", len(d) > 0 and abs(float(np.mean(d)) - 20.0) < 0.5 and float(np.max(d)) < 30.0,
              "평균 %.2f ms, 최대 %.1f ms" % (float(np.mean(d)), float(np.max(d))))
        up = next((x[0] for x in T if abs((x[2] / 1000.0) % 360.0 - 180.0) < 10.0), None)
        print("      처음 직립 근처에 온 때 %.2f s, 팔 이동 최대 %.0f°, 힘 최대 %d/1000" % (
            (up or -1) / 1000.0, max(abs(x[1]) for x in T) / 1000.0, max(abs(x[5]) for x in T)))

        time.sleep(4.0)
        r, T, M, _ = c.run(10, stop_at=2.0)
        check("정지 명령", r is not None and r.startswith("R END stop") and 80 < len(T) < 130, "%s, 상태 줄 %d개" % (r, len(T)))
        time.sleep(4.0)
        r, T, M, _ = c.run(10, alive=False)
        check("살아 있음 신호 없이 시작", r is not None and r.startswith("R END link") and len(T) < 110, "%s, 상태 줄 %d개" % (r, len(T)))
        time.sleep(4.0)
        r, T, M, _ = c.run(10, alive_until=1.5)
        check("실행 중 신호 끊김 -> 2초 뒤 정지", r is not None and r.startswith("R END link") and 150 < len(T) < 210,
              "%s, 상태 줄 %d개" % (r, len(T)))
        r = c.ask("Z")
        check("모르는 명령", r == "E cmd Z", r)
        r = c.ask("C", 5.0)
        check("점검", r is not None and r.startswith("C OK"), r)
        c.close()
    finally:
        out = finish(sv)
    print("--- 서버 출력")
    print("\n".join("    " + x for x in out.splitlines() if x.strip()))

    sv = server(host_dir, port + 1, ["--fake-swing-deg", "40"])
    try:
        c = Client(port + 1)
        c.upload(raw)
        r, T, M, lead = c.run(8)
        waits = [m for m in M if m.startswith("M still ")]
        check("흔들리는 진자 기다렸다가 시작", r is not None and r.startswith("R END done") and len(waits) >= 2 and stood(T),
              "%s, 기다림 안내 %d번, 시작까지 %.1f s" % (r, len(waits), lead or -1))
        c.close()
    finally:
        finish(sv)

    sv = server(host_dir, port + 3, ["--fake-tilt-deg", "2.0", "--fake-tilt-dir", "100", "--fake-zero-shift", "5"])
    try:
        c = Client(port + 3)
        r = c.ask("L")
        check("맞추기 값 없음 (처음)", r is not None and r.startswith("L NONE"), r)
        r, mid, secs = c.cal(stop_at=6.0)
        check("키트 맞추기 중 정지 명령", r.startswith("G FAIL stop"), "%s (%.1f s)" % (r, secs))
        time.sleep(3.0)
        r, mid, secs = c.cal()
        k = kv(r)
        J_true = 0.0245 + 2.358e-4
        check("키트 맞추기 (받침 2.0°/100°, 영점 +5)", r.startswith("G OK") and abs(k["tilt"] - 2.0) < 0.2 and abs(k["dir"] - 100) < 10
              and abs(k["C"] - (2033.9 + 5)) < 2.0, "%s (%.1f s)" % (r, secs))
        check("팔 관성·마찰 (가짜 키트 참값 J %.4f, Fc 0.69/0.68, b 0.17)" % J_true,
              r.startswith("G OK") and abs(k["J"] / J_true - 1) < 0.03 and abs(k["Fp"] - 0.69) < 0.03 and abs(k["Fm"] - 0.68) < 0.03
              and abs(k["b"] - 0.17) < 0.02, "J %s Fp %s Fm %s b %s" % (k.get("J"), k.get("Fp"), k.get("Fm"), k.get("b")))
        check("진자 주기 (모델과 같아야 함)", r.startswith("G OK") and k.get("T0") and abs(k["T0"] / k["T0m"] - 1) < 0.01,
              "T0 %s / 모델 %s" % (k.get("T0"), k.get("T0m")))
        r2 = c.ask("L")
        check("맞춘 값 다시 읽기", r2 is not None and r2.split()[1] == r.split()[2], r2)
        c.upload(raw)
        r, T, M, lead = c.run(8)
        lv = [m for m in M if m.startswith("M level ok")]
        tl = kv(lv[0]).get("tilt", 9) if lv else 9
        check("실행마다 수평 측정 (받침 2°) 뒤 세움", r.startswith("R END done") and abs(tl - 2.0) < 0.4 and stood(T),
              "%s, %s, 시작까지 %.1f s" % (r, lv[0] if lv else "-", lead or -1))
        c.close()
    finally:
        finish(sv)

    sv = server(host_dir, port + 5, ["--fake-tilt-deg", "1.0", "--fake-tilt-dir", "200"])
    try:
        c = Client(port + 5)
        c.upload(raw)
        r, T, M, _ = c.run(8)
        check("맞춘 영점(+5)과 진자 영점이 다르면 시작 안 함", r.startswith("R END level") and not T, r)
        c.close()
    finally:
        finish(sv)
    os.remove(os.path.join(LOGS, "kit_cal_latest.json"))

    sv = server(host_dir, port + 4, ["--fake-tilt-deg", "6.0", "--fake-tilt-dir", "30"])
    try:
        c = Client(port + 4)
        c.upload(raw)
        r, T, M, _ = c.run(8)
        check("받침 6° 이면 시작 안 함", r.startswith("R END tilt") and not T, r)
        c.close()
    finally:
        out = finish(sv)
    stops = [x for x in out.splitlines() if "종료:" in x]
    print("    " + (stops[-1] if stops else "(정지 순서 출력 없음)"))

    sv = server(host_dir, port + 2, ["--fake-temp", "70"])
    try:
        c = Client(port + 2)
        c.upload(raw)
        r, T, M, _ = c.run(8)
        check("모터 과열이면 시작 안 함", r is not None and r.startswith("R END hot") and not T, r)
        c.close()
    finally:
        out = finish(sv)
    stops = [x for x in out.splitlines() if "종료:" in x]
    print("    " + (stops[-1] if stops else "(정지 순서 출력 없음)"))

    print("\n%d / %d 통과" % (sum(res), len(res)))
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
