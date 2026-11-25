"""직렬 포트 경로 시험 (Pi·리눅스 전용): 가짜 직렬 포트(pty)를 만들어 kit_server.py --fake --serial 에 붙입니다.

  python3 test_serial_pty.py <host-dir> <정책.npz> [초=10]

실제 USB 케이블 대신 pty 를 쓰는 것만 다르고, 대기 프로그램의 직렬 열기(termios raw)·읽기·쓰기는 그대로 탑니다.
can0, I2C, 모터를 쓰지 않습니다. Pi 에서 돌리면 제어 주기(20 ms)가 지켜지는지도 함께 봅니다.
"""
import os
import pty
import sys
import time
import zlib
import base64
import select
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    host_dir, npz = sys.argv[1], sys.argv[2]
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    w = np.load(npz)
    raw = np.concatenate([np.asarray(w[k], dtype="<f4").ravel() for k in (
        "latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "mu.weight", "mu.bias")]).tobytes()
    crc = "%08x" % (zlib.crc32(raw) & 0xFFFFFFFF)
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    sv = subprocess.Popen([sys.executable, os.path.join(HERE, "kit_server.py"), "--fake", "--serial", path,
                           "--host-dir", host_dir, "--log-dir", "/tmp/kit_fake_logs"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    buf = b""

    def tx(line):
        os.write(master, (line + "\n").encode())

    def rx(timeout):
        nonlocal buf
        end = time.time() + timeout
        while b"\n" not in buf:
            left = end - time.time()
            if left <= 0 or not select.select([master], [], [], left)[0]:
                return None
            buf += os.read(master, 4096)
        line, buf = buf.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()

    ok = True
    try:
        time.sleep(1.5)
        tx("H")
        r = rx(3.0)
        print("인사:", r)
        ok &= bool(r and r.startswith("I furuta-rev11") and " v2" in r)
        t0 = time.time()
        tx("P BEGIN %d %s relu 1 6,64,64,1" % (len(raw) // 4, crc))
        b = base64.b64encode(raw).decode()
        for i in range(0, len(b), 180):
            tx("P " + b[i:i + 180])
        tx("P END")
        r = rx(10.0)
        print("정책 올리기:", r, "(%.2f s)" % (time.time() - t0))
        ok &= r == "P OK " + crc
        tx("R %g pty" % secs)
        T, host_t, last_a, end = [], [], 0.0, None
        t0 = time.time()
        while time.time() - t0 < secs + 30:
            if time.time() - last_a > 0.5:
                tx("A")
                last_a = time.time()
            line = rx(0.1)
            if line is None:
                continue
            if line.startswith("T "):
                T.append([int(x) for x in line.split()[1:]])
                host_t.append(time.time())
            elif line.startswith("R END"):
                end = line
                break
            else:
                print("   ", line)
        print("끝:", end, "/ 상태 줄", len(T))
        ok &= bool(end and end.startswith("R END done"))
        d = np.diff([x[0] for x in T])
        g = np.diff(host_t) * 1000.0
        print("키트 시각 간격 ms: 평균 %.2f, 표준편차 %.2f, 최대 %d, 22 ms 넘은 주기 %d / %d" % (
            d.mean(), d.std(), d.max(), int((d > 22).sum()), len(d)))
        print("받는 쪽 도착 간격 ms: 평균 %.2f, 최대 %.1f" % (g.mean(), g.max()))
        tail = T[-100:]
        up = all(abs((x[2] / 1000.0) % 360.0 - 180.0) < 10.0 for x in tail)
        print("마지막 2초 직립 ±10°:", up)
        ok &= up and abs(d.mean() - 20.0) < 0.5 and d.max() <= 30
    finally:
        sv.terminate()
        out = sv.communicate(timeout=10)[0].decode("utf-8", "replace")
        print("--- 서버 출력")
        print("\n".join("    " + x for x in out.splitlines() if x.strip()))
    print("통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
