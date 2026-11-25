/* 환경(environment) 연결 계층.
 *
 * 앱의 나머지 코드는 "장치"에게 상태를 받고 모터 힘을 보낼 뿐, 그것이 화면 속 시뮬레이션인지
 * 책상 위 키트인지 거의 구분하지 않습니다. 다른 점은 하나입니다: 시뮬레이션은 앱이 매 순간 정책을 돌리고,
 * 실제 키트는 정책을 통째로 받아 키트 안에서 돌립니다(USB 지연이 제어에 끼어들지 않게).
 *
 * 상태는 모두 같은 모양입니다: { t, th, al, thd, ald, u, flags }
 *   th, al  : 팔·진자 각도 [rad], al = 0 이 아래로 매달림
 *   thd, ald: 각속도 [rad/s]
 *   u       : 실제로 들어간 모터 힘 (−1…1)
 *
 * 줄 단위 규약(v2)은 app/PROTOCOL.md, 키트 쪽 프로그램은 app/kit/kit_server.py 입니다.
 */
(function (PS) {
  "use strict";

  function Emitter() { this._h = {}; }
  Emitter.prototype.on = function (k, fn) { (this._h[k] = this._h[k] || []).push(fn); return this; };
  Emitter.prototype.emit = function (k, a) {
    (this._h[k] || []).forEach(function (fn) { fn(a); });
  };

  /* ── 화면 속 환경 ── */
  function SimDevice(C) {
    Emitter.call(this);
    this.kind = "sim";
    this.id = "sim";
    this.name = "시뮬레이션";
    this.C = C;
    this.sim = new PS.Furuta(C);
    this.connected = true;
    this.enabled = true;
    this.ctrlDt = this.sim.ctrlDt;
  }
  SimDevice.prototype = Object.create(Emitter.prototype);
  SimDevice.prototype.connect = function () { this.connected = true; this.emit("status", this); return Promise.resolve(); };
  SimDevice.prototype.disconnect = function () { return Promise.resolve(); };
  SimDevice.prototype.enable = function () { this.enabled = true; };
  SimDevice.prototype.disable = function () { this.enabled = false; };
  SimDevice.prototype.zero = function () { };
  SimDevice.prototype.reset = function (th, al, thd, ald) {
    this.sim.reset(th, al, thd, ald);
    return this.state();
  };
  SimDevice.prototype.state = function () {
    var s = this.sim;
    return { t: s.t, th: s.th, al: s.al, thd: s.thd, ald: s.ald, u: s.u, flags: s.dead ? 2 : 0 };
  };
  SimDevice.prototype.command = function (u) {
    this.sim.step(this.enabled ? u : 0);
    return this.state();
  };
  SimDevice.prototype.obs = function (out) { return this.sim.obs(out); };

  /* ── 실제 키트 (규약 v2, app/PROTOCOL.md) ──
   * 정책은 키트 안의 라즈베리파이가 50 Hz 로 직접 돌립니다. 앱은 가중치를 올리고(upload), 실행을 시키고(run),
   * 키트가 보내 주는 상태 줄로 그림과 그래프를 그립니다. USB 지연이 제어 지연이 되지 않습니다. */
  var CRC_T = null;
  function crc32(bytes) {
    var i, k, c;
    if (!CRC_T) {
      CRC_T = new Uint32Array(256);
      for (i = 0; i < 256; i++) {
        c = i;
        for (k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
        CRC_T[i] = c >>> 0;
      }
    }
    c = 0xFFFFFFFF;
    for (i = 0; i < bytes.length; i++) c = CRC_T[(c ^ bytes[i]) & 255] ^ (c >>> 8);
    return ((c ^ 0xFFFFFFFF) >>> 0).toString(16).padStart(8, "0");
  }

  /* 정책을 키트로 보낼 줄들로 바꿉니다. 값은 층마다 W(행 우선) 다음 b, float32 little-endian, base64. */
  function packPolicy(mlp) {
    var v = mlp.flatten(), dv = new DataView(new ArrayBuffer(v.length * 4)), i;
    for (i = 0; i < v.length; i++) dv.setFloat32(i * 4, v[i], true);
    var bytes = new Uint8Array(dv.buffer), crc = crc32(bytes), bin = "";
    for (i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    var b64 = btoa(bin);
    var sizes = [mlp.layers[0].cols].concat(mlp.layers.map(function (L) { return L.rows; }));
    var lines = ["P BEGIN " + v.length + " " + crc + " " + (mlp.act === Math.tanh ? "tanh" : "relu") + " " +
      (mlp.squash ? 1 : 0) + " " + sizes.join(",")];
    for (i = 0; i < b64.length; i += 180) lines.push("P " + b64.slice(i, i + 180));
    lines.push("P END");
    return { crc: crc, lines: lines, bytes: bytes.length };
  }

  var ZERO = { t: 0, th: 0, al: 0, thd: 0, ald: 0, u: 0, flags: 0 };

  /* 키트 맞추기 값 줄: "L <stamp|NONE> k=v ..." 또는 "G OK <stamp> k=v ..." 의 OK 뒤부터. */
  function parseCal(p) {
    var out = { none: p[1] === "NONE", stamp: p[1] === "NONE" ? null : p[1] };
    p.slice(2).forEach(function (s) {
      var m = s.split("=");
      if (m.length === 2) out[m[0]] = m[1] === "-" ? null : +m[1];
    });
    return out;
  }

  function KitDevice(portPath) {
    Emitter.call(this);
    this.kind = "kit";
    this.id = portPath;
    this.name = portPath;
    this.port = portPath;
    this.connected = false;
    this.info = null;
    this.ctrlDt = 0.02;
    this.last = ZERO;
    this.polCrc = null;
    this.lastEnd = null;
    this.running = false;
    this._wait = [];
    this._h = this._h || {};
    this._run = null;
    this._cal = null;
    this._alive = null;
    this.cal = null;
    this.level = null;
  }
  KitDevice.prototype = Object.create(Emitter.prototype);

  KitDevice.installBridge = function () {
    if (KitDevice._wired || !window.studio) return;
    KitDevice._wired = true;
    window.studio.serial.onLine(function (line) {
      if (KitDevice.current) KitDevice.current._line(line);
    });
    window.studio.serial.onClosed(function () {
      var d = KitDevice.current;
      if (d) d._lost("연결이 끊어졌습니다");
    });
    window.studio.serial.onError(function (msg) {
      var d = KitDevice.current;
      if (d) d.emit("alert", { code: "error", message: msg });
    });
  };

  KitDevice.prototype._write = function (line) {
    return window.studio.serial.write(line);
  };

  /* 조건에 맞는 줄이 올 때까지 기다립니다. */
  KitDevice.prototype._expect = function (test, ms, what) {
    var self = this;
    return new Promise(function (resolve, reject) {
      var w = { test: test, resolve: resolve };
      w.timer = setTimeout(function () {
        self._wait = self._wait.filter(function (x) { return x !== w; });
        reject(new Error(what + ": 키트가 응답하지 않습니다"));
      }, ms);
      self._wait.push(w);
    });
  };

  KitDevice.prototype._lost = function (message) {
    this.connected = false;
    this._endCal(null, message);
    this._endRun({ reason: "link", steps: 0, detail: message });
    this.emit("status", this);
    this.emit("alert", { code: "closed", message: message });
  };

  KitDevice.prototype._endRun = function (end) {
    var r = this._run;
    if (this._alive) { clearInterval(this._alive); this._alive = null; }
    if (this._limit) { clearTimeout(this._limit); this._limit = null; }
    this._run = null;
    this.running = false;
    this.lastEnd = end.reason;
    if (r && r.onEnd) r.onEnd(end);
    this.emit("status", this);
  };

  KitDevice.prototype._line = function (line) {
    var p = line.trim().split(/\s+/), c = p[0], D = Math.PI / 180, r = this._run;
    if (!c) return;
    if (c === "T") {
      this.last = {
        t: +p[1] / 1000, th: +p[2] / 1000 * D, al: +p[3] / 1000 * D,
        thd: +p[4] / 1000 * D, ald: +p[5] / 1000 * D, u: +p[6] / 1000, flags: +p[7] | 0
      };
      if (r && r.onState) r.onState(this.last);
      return;
    }
    if (c === "I") {
      this.info = { name: p[1], fw: p[2], hz: +p[3] || 50, tau_mNm: +p[4] || 0, v2: p[5] === "v2", fake: p[6] === "fake" };
      this.name = p[1] || this.port;
      this.ctrlDt = 1 / this.info.hz;
    } else if (c === "K") {
      this.polCrc = p[2] && p[2] !== "-" ? p[2] : null;
      this.lastEnd = p[3] && p[3] !== "-" ? p[3] : null;
    } else if (c === "P" && p[1] === "OK") {
      this.polCrc = p[2];
    } else if (c === "M") {
      if (p[1] === "level" && p[2] === "ok") this.level = parseCal(["", "run"].concat(p.slice(3)));
      if (r && r.onMsg) r.onMsg(p[1], p.slice(2));
      if (this._cal && this._cal.onMsg) this._cal.onMsg(p[1], p.slice(2));
    } else if (c === "G") {
      if (p[1] === "OK") this._endCal(parseCal(p.slice(1)));
      else if (p[1] === "FAIL") this._endCal(null, (p[2] || "error") + " " + p.slice(3).join(" "));
      else if (this._cal && this._cal.onMsg) this._cal.onMsg(p[1], p.slice(2));
    } else if (c === "L") {
      this.cal = parseCal(p);
    } else if (c === "R" && p[1] === "START") {
      if (r && r.onMsg) r.onMsg("start", []);
    } else if (c === "R" && p[1] === "END") {
      this._endRun({ reason: p[2] || "error", steps: +p[3] || 0, detail: p.slice(4).join(" ") });
    } else if (c === "E" || c === "!") {
      this.emit("alert", { code: p[1] || "", message: p.slice(2).join(" ") });
    }
    var hit = this._wait.filter(function (w) { return w.test(p); });
    if (hit.length) {
      this._wait = this._wait.filter(function (w) { return hit.indexOf(w) < 0; });
      hit.forEach(function (w) { clearTimeout(w.timer); w.resolve(p); });
    }
  };

  KitDevice.prototype.connect = function () {
    var self = this;
    KitDevice.installBridge();
    KitDevice.current = this;
    return window.studio.serial.open(this.port, 115200).then(function () {
      return new Promise(function (resolve, reject) {
        var tries = 0;
        self._write("H");
        var iv = setInterval(function () {
          if (self.info) { clearInterval(iv); resolve(); return; }
          if (++tries > 16) { clearInterval(iv); reject(new Error("키트가 응답하지 않습니다. 키트 전원과 케이블을 확인하세요")); return; }
          self._write("H");
        }, 250);
      });
    }).then(function () {
      if (!self.info.v2) throw new Error("키트 프로그램이 오래된 버전입니다 (규약 v2 필요)");
      var w = self._expect(function (p) { return p[0] === "K"; }, 2000, "상태 읽기");
      self._write("Q");
      return w;
    }).then(function (p) {
      if (p[1] !== "idle") throw new Error("키트가 다른 실행을 하는 중입니다. 잠시 뒤에 다시 연결하세요");
      self.connected = true;
      self.emit("status", self);
    }).catch(function (e) {
      window.studio.serial.close();
      if (KitDevice.current === self) KitDevice.current = null;
      throw e;
    });
  };

  KitDevice.prototype.disconnect = function () {
    var self = this;
    return Promise.resolve(this.running ? this._write("X") : null).then(function () {
      return window.studio.serial.close();
    }).then(function () {
      self.connected = false;
      if (KitDevice.current === self) KitDevice.current = null;
      self._endRun({ reason: "link", steps: 0, detail: "연결을 끊었습니다" });
    });
  };

  /* 정책 올리기. 키트가 이미 같은 정책(CRC)을 갖고 있으면 보내지 않습니다. */
  KitDevice.prototype.upload = function (mlp) {
    var self = this, pk = packPolicy(mlp);
    if (this.polCrc === pk.crc) return Promise.resolve(pk.crc);
    var w = this._expect(function (p) {
      return (p[0] === "P" && p[1] === "OK") || (p[0] === "E" && p[1] === "policy");
    }, 20000, "정책 올리기");
    this._write(pk.lines);
    return w.then(function (p) {
      if (p[0] === "E") throw new Error("키트가 정책을 받지 않았습니다 (" + p.slice(2).join(" ") + ")");
      if (p[2] !== pk.crc) throw new Error("키트에 올라간 정책이 보낸 것과 다릅니다");
      self.polCrc = p[2];
      return p[2];
    });
  };

  /* 키트에서 secs 초 동안 실행합니다. h: { onMsg(code, args), onState(st), onEnd({reason, steps, detail}) } */
  KitDevice.prototype.run = function (secs, h) {
    var self = this;
    this._run = h || {};
    this.running = true;
    this.last = ZERO;
    this._write("R " + secs);
    this._write("A");
    this._alive = setInterval(function () { self._write("A"); }, 500);
    this._limit = setTimeout(function () {
      self._endRun({ reason: "link", steps: 0, detail: "키트가 끝났다고 알려 오지 않았습니다" });
    }, (secs + 60) * 1000);
    this.emit("status", this);
  };

  KitDevice.prototype.stop = function () {
    if (this.running) this._write("X");
  };

  /* 키트 점검: 모터를 움직이지 않고 CAN·모터 응답·진자 센서만 읽습니다. */
  KitDevice.prototype.check = function () {
    var w = this._expect(function (p) { return p[0] === "C"; }, 8000, "키트 점검");
    this._write("C");
    return w.then(function (p) {
      var out = { ok: p[1] === "OK", text: p.slice(2).join(" ") };
      if (out.ok) p.slice(2).forEach(function (kv) { var m = kv.split("="); out[m[0]] = +m[1]; });
      return out;
    });
  };

  /* 키트에 남아 있는 마지막 키트 맞추기 값 (모터를 움직이지 않음). */
  KitDevice.prototype.calInfo = function () {
    var self = this, w = this._expect(function (p) { return p[0] === "L"; }, 3000, "맞추기 값 읽기");
    this._write("L");
    return w.then(function () { return self.cal; });
  };

  /* 키트 맞추기 (모터가 움직입니다). 사용자가 확인 단추를 눌렀을 때만 부릅니다.
   * h: { onMsg(code, args) }. 끝나면 값({stamp, C, tilt, dir, J, Jm, Fp, Fm, b, T0, T0m, temp, ok}). */
  KitDevice.prototype.calibrate = function (h) {
    var self = this;
    if (this.running) return Promise.reject(new Error("키트가 다른 일을 하는 중입니다"));
    return new Promise(function (resolve, reject) {
      self._cal = { onMsg: (h || {}).onMsg, resolve: resolve, reject: reject };
      self.running = true;
      self._write("G");
      self._write("A");
      self._alive = setInterval(function () { self._write("A"); }, 500);
      self._limit = setTimeout(function () { self._endCal(null, "키트가 끝났다고 알려 오지 않았습니다"); }, 180000);
      self.emit("status", self);
    });
  };

  KitDevice.prototype._endCal = function (cal, why) {
    var c = this._cal;
    if (!c) return;
    this._cal = null;
    if (this._alive) { clearInterval(this._alive); this._alive = null; }
    if (this._limit) { clearTimeout(this._limit); this._limit = null; }
    this.running = false;
    if (cal) { this.cal = cal; c.resolve(cal); } else c.reject(new Error(why || "키트 맞추기 실패"));
    this.emit("status", this);
  };

  KitDevice.prototype.state = function () { return this.last; };
  KitDevice.packPolicy = packPolicy;

  /* ── 한 번의 시험을 돌립니다. 시뮬레이션은 화면 주사율에 맞춰 밀고,
   *    실제 키트는 opt.policy 를 올린 뒤 장치가 보내 주는 상태에 맞춰 따라갑니다.
   *    끝나면 onDone(마지막 상태, end). end 는 키트일 때만 있고 { reason, steps, detail } 입니다. ── */
  function runEpisode(dev, controller, opt) {
    opt = opt || {};
    var secs = opt.seconds || 10;
    var onTick = opt.onTick || function () { };
    var onDone = opt.onDone || function () { };
    var obs = new Float64Array(PS.OBS_N);
    var stop = false;

    if (dev.kind === "sim") {
      var prev = performance.now(), acc = 0, speed = opt.speed || 1;
      dev.reset(opt.th0 || 0, opt.al0 !== undefined ? opt.al0 : 0, 0, 0);
      /* 타이머로 밉니다. requestAnimationFrame 은 창이 가려지면 멈추거나 아주 느려져서 시험이 끝나지 않습니다.
       * 가려진 창은 타이머도 1초에 한 번으로 늦춰지므로, 밀린 시간을 1.2초까지 한 번에 따라잡게 둡니다. */
      var timer = setTimeout(function frame() {
        var now = performance.now();
        acc += Math.min(1.2, (now - prev) / 1000) * speed;
        prev = now;
        while (acc >= dev.ctrlDt && dev.sim.t < secs && !stop && !dev.sim.dead) {
          dev.obs(obs);
          dev.command(controller(obs, dev.sim));
          acc -= dev.ctrlDt;
        }
        var st = dev.state();
        onTick(st);
        if (!stop && st.t < secs && !dev.sim.dead) timer = setTimeout(frame, 16);
        else if (!stop) onDone(st);
      }, 16);
      return { cancel: function () { stop = true; clearTimeout(timer); } };
    }

    /* 실제 키트: 정책을 올리고, 실행을 시키고, 키트가 보내 주는 상태를 따라갑니다. */
    var ended = false;
    var fin = function (end) {
      if (ended) return;
      ended = true;
      if (!stop) onDone(dev.last, end);
    };
    var onMsg = opt.onMsg || function () { };
    onMsg("upload", []);
    dev.upload(opt.policy).then(function () {
      if (stop) return;
      dev.run(secs, {
        onMsg: function (code, args) { if (!stop) onMsg(code, args); },
        onState: function (st) { if (!stop) onTick(st); },
        onEnd: fin
      });
    }).catch(function (e) {
      fin({ reason: "error", steps: 0, detail: String(e && e.message || e) });
    });
    return { cancel: function () { stop = true; dev.stop(); } };
  }

  PS.SimDevice = SimDevice;
  PS.KitDevice = KitDevice;
  PS.runEpisode = runEpisode;
})(window.PS = window.PS || {});
