"""명령줄 클라이언트: 앱 없이 kit_server.py 에 붙어 점검·정책 올리기·실행을 합니다 (TCP).

  python3 kit_client.py --check                              # 점검만 (모터 안 움직임)
  python3 kit_client.py --policy <정책.npz> --run 10         # 올리고 10초 실행 (모터가 움직임!)

실행 중 Ctrl-C 를 누르면 정지(X)를 보냅니다. 상태 줄은 --save <파일.csv> 로 남길 수 있습니다.
"""
import sys
import time
import zlib
import base64
import socket
import argparse

import numpy as np

KEYS = ("latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias")


class Client:
    def __init__(self, host, port):
        self.s = socket.create_connection((host, port), timeout=3.0)
        self.buf = b""

    def tx(self, line):
        self.s.sendall((line + "\n").encode("utf-8"))

    def rx(self, timeout):
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
        return line.decode("utf-8", "replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--run", type=float, default=0.0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    c = Client(a.host, a.port)
    c.tx("H")
    print("인사:", c.rx(3.0))
    c.tx("Q")
    print("상태:", c.rx(3.0))
    if a.check:
        c.tx("C")
        print("점검:", c.rx(10.0))
    if a.policy:
        w = np.load(a.policy)
        raw = np.concatenate([np.asarray(w[k], dtype="<f4").ravel() for k in KEYS]).tobytes()
        crc = "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
        c.tx("P BEGIN %d %s relu 1 6,64,64,1" % (len(raw) // 4, crc))
        b = base64.b64encode(raw).decode()
        for i in range(0, len(b), 180):
            c.tx("P " + b[i:i + 180])
        c.tx("P END")
        r = c.rx(10.0)
        print("정책 올리기:", r)
        if r != "P OK " + crc:
            return 1
    if a.run > 0:
        c.tx("R %g kit_client" % a.run)
        T, last_a, end, t0 = [], 0.0, None, time.time()
        try:
            while time.time() - t0 < a.run + 90:
                if time.time() - last_a > 0.5:
                    c.tx("A")
                    last_a = time.time()
                line = c.rx(0.1)
                if line is None:
                    continue
                if line.startswith("T "):
                    T.append([int(x) for x in line.split()[1:]])
                elif line.startswith("R END"):
                    end = line
                    break
                else:
                    print("   ", line, flush=True)
        except KeyboardInterrupt:
            c.tx("X")
            end = c.rx(5.0)
        print("끝:", end, "/ 상태 줄", len(T))
        if a.save and T:
            with open(a.save, "w") as fh:
                fh.write("t_ms,theta_mdeg,alpha_mdeg,theta_mdps,alpha_mdps,u_applied,flags\n")
                fh.writelines(",".join(str(v) for v in r) + "\n" for r in T)
        if T:
            phi = [abs((x[2] / 1000.0) % 360.0 - 180.0) for x in T]
            up, held = None, 0
            for i, p in enumerate(phi):
                held = held + 1 if p < 10.0 else 0
                if held == 25:
                    up = T[i - 24][0] / 1000.0
                    break
            d = np.diff([x[0] for x in T])
            tail = phi[-100:]
            print("  세운 때 %s, 마지막 2초 최대 기울기 %.1f°, 팔 최대 이동 %.0f°, 팔 최고 속도 %.0f dps, 진자 최고 속도 %.1f rad/s" % (
                "%.2f s" % up if up is not None else "없음", max(tail), max(abs(x[1]) for x in T) / 1000.0,
                max(abs(x[3]) for x in T) / 1000.0, max(abs(x[4]) for x in T) / 1000.0 * 3.14159 / 180))
            if len(d):
                print("  주기 ms: 평균 %.2f, 최대 %d, 22 넘은 주기 %d / %d, 제동 제한 걸린 주기 %d" % (
                    d.mean(), d.max(), int((d > 22).sum()), len(d), sum(1 for x in T if x[6] & 32)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
