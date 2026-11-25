/* 화면 흐름과 상태.
 *
 * 화면은 하나입니다(엔트리 배치). 왼쪽에 실행 화면·시작하기·보조 창, 오른쪽에 살펴보기·블록·그래프·예시 규칙.
 * 활동 순서(1 살펴보기 → 2 점수 규칙 만들기 → 3 가르치고 시험하기)는 화면을 바꾸지 않고 지금 할 일만 알려 줍니다.
 *
 * 시작하기를 누르면 '시작하기 버튼을 클릭했을 때' 아래 블록을 위에서부터 실행합니다.
 *   · 따라 하기: 고르는 칸 아홉 조합마다 MuJoCo에서 미리 학습해 둔 정책을 불러오고, 그때 기록된 점수 곡선을 보여 줍니다.
 *   · 자유롭게 만들기: 가장 가까운 정책에서 출발해 화면 안에서 CEM으로 고쳐 배웁니다(몇 초~수십 초).
 * 실제 키트에서는 같은 정책이 시뮬레이션에서 진자를 세운 뒤에만 시험할 수 있습니다.
 */
(function (PS) {
  "use strict";

  var C = PS.MODEL;
  var GT = C ? C.gate.conds.length * C.gate.starts.length : 0;
  var $ = function (q) { return document.querySelector(q); };
  var id = function (x) { return document.getElementById(x); };

  var st = {
    mode: "guide", smooth: 1, effort: 1, env: "sim", tab: "intro", step: 1, done1: false,
    policy: null, policyKey: null, simOK: false, learnedSec: null, gate: null, fallback: null,
    busy: false, run: null, trainer: null, anim: null,
    kit: null, kitReady: false, runNo: 0, row: null, rowKey: null, prev: null
  };

  var simDev = new PS.SimDevice(C);
  var stage = new PS.Stage($('[data-stage="a"]'), { nums: true });
  var blocks = new PS.Blocks(id("boardHost"), id("palette"), id("rail"), {
    onChange: function () { changed(); },
    onPick: function (name, tip) {
      id("helpName").textContent = name;
      id("helpText").textContent = tip || "이 블록의 설명은 아직 없습니다.";
    },
    onGuide: function (type, lv) {
      if (type === "g_smooth") st.smooth = +lv;
      if (type === "g_effort") st.effort = +lv;
    }
  });

  /* ── 그래프: 이번 시도와 지난번 시도를 겹쳐 봅니다 ── */
  var CUR = "#3b6de4", PREV = "#8a8880";
  var charts = {
    learn: new PS.Chart(id("chLearn"), { xlabel: "세대", digits: 0 }),
    height: new PS.Chart(id("chHeight"), { xlabel: "초", ymin: 0, ymax: 100, digits: 0 }),
    force: new PS.Chart(id("chForce"), { xlabel: "초", ymin: 0, ymax: 100, digits: 0 }),
    arm: new PS.Chart(id("chArm"), { xlabel: "초", ymin: 0, digits: 1 })
  };
  function newRun() { return { label: "", curve: [], t: [], height: [], force: [], arm: [], secs: 10 }; }
  st.runData = newRun();
  st.rows = [];
  st.curRow = null;
  st.pin = null;
  /* 비교 대상: 기록 표에서 눌러 고른 시도, 없으면 이번 시도 바로 앞의 시도 */
  function compareRow() {
    if (st.pin && st.pin !== st.curRow) return st.pin;
    for (var i = 0; i < st.rows.length; i++) if (st.rows[i] !== st.curRow) return st.rows[i];
    return null;
  }
  function drawCharts(which) {
    var c = st.runData, cr = compareRow(), p = cr ? cr.data : null;
    function ser(xc, yc, xp, yp) {
      var out = [];
      if (p && yp.length) out.push({ name: "비교", color: PREV, dash: "5 4", x: xp, y: yp });
      out.push({ name: "이번", color: CUR, x: xc, y: yc });
      return out;
    }
    function idx(a) { return a.map(function (_v, i) { return i + 1; }); }
    if (!which || which === "learn") charts.learn.set(ser(idx(c.curve), c.curve, p ? idx(p.curve) : [], p ? p.curve : []));
    if (!which || which === "test") {
      var xm = Math.max(c.secs, p && p.t.length ? p.secs : 0);
      charts.height.set(ser(c.t, c.height, p ? p.t : [], p ? p.height : []), xm);
      charts.force.set(ser(c.t, c.force, p ? p.t : [], p ? p.force : []), xm);
      charts.arm.set(ser(c.t, c.arm, p ? p.t : [], p ? p.arm : []), xm);
    }
    id("gPrev").hidden = !p;
    id("gNote").textContent = cr ? "비교: " + cr.no + "회 · " + cr.label + " (왼쪽 기록 표에서 바꿀 수 있습니다)"
      : "한 번 더 시도하면 앞의 시도와 겹쳐 보입니다";
  }
  window.addEventListener("resize", function () { drawCharts(); });

  var HINT = {
    1: function () { return "미리 배운 에이전트를 실행해 보고, 학습지 2쪽에 본 것을 적습니다"; },
    2: function () {
      return st.mode === "guide"
        ? "블록 안의 고르는 칸(▼)을 바꿔 규칙을 정하고, 학습지 3쪽에 예상을 적습니다"
        : "블록을 끌어다 점수 규칙을 만들고, 학습지 3쪽에 예상을 적습니다";
    },
    3: function () { return "시작하기를 누르면 배우고 나서 시험합니다. 결과를 학습지 3쪽에 옮겨 적습니다"; }
  };

  /* ── 규칙 ── */
  function rule() {
    return st.mode === "guide" ? PS.reward.basic(st.smooth, st.effort) : blocks.rule();
  }
  function runOpts() {
    if (st.mode === "guide") return { gens: 24, pop: 20, sigma: 0.06, learn: true, test: 10 };
    return blocks.runOpts();
  }
  function keyOf(r, o) {
    return st.mode === "guide" ? r.key : r.key + "|" + o.gens + "|" + o.pop + "|" + o.sigma;
  }
  function say(text) { id("runMsg").textContent = text || ""; }

  /* ── 결과 판정: 학습지 3쪽의 보기와 같은 말로 ── */
  function Judge() {
    this.up = null; this.fell = null; this.held = 0; this.wob = 0; this.peak = 0;
    this.fsum = 0; this.fn = 0; this.th0 = null; this.th1 = 0;
  }
  Judge.prototype.tick = function (s, phi, dt) {
    var deg = Math.abs(phi) * 180 / Math.PI, t = s.t;
    if (s.flags & 2) this.stopped = true;
    if (this.th0 === null) this.th0 = s.th;
    this.th1 = s.th;
    this.peak = Math.max(this.peak, 180 - deg);
    if (deg < 17 && Math.abs(s.ald) < 3) {
      this.held += dt;
      if (this.up === null && this.held > 0.4) this.up = t - 0.4;
    } else this.held = 0;
    if (this.up !== null && this.fell === null) {
      if (deg > 45) this.fell = t;
      else if (t > this.up + 1) {
        this.wob = Math.max(this.wob, deg);
        this.fsum += Math.abs(s.u);
        this.fn++;
      }
    }
  };
  /* 키트가 알려 준 끝난 까닭 (규약 v2 의 R END) */
  var KITEND = {
    stop: ["정지했습니다", "정지"],
    pend_speed: ["안전 정지 (너무 빨리 돌았습니다)", "안전 정지"],
    arm_speed: ["안전 정지 (너무 빨리 돌았습니다)", "안전 정지"],
    spin: ["안전 정지 (진자가 계속 돌았습니다)", "안전 정지(회전)"],
    not_still: ["시작 못 함 (진자가 아래로 멈추지 않았습니다)", "시작 못 함"],
    check: ["시작 못 함 (키트 점검에서 걸렸습니다)", "시작 못 함"],
    hot: ["시작 못 함 (모터가 뜨겁습니다. 식힌 뒤 다시 하세요)", "모터 과열"],
    link: ["연결이 끊겨 키트가 멈췄습니다", "연결 끊김"],
    watchdog: ["모터가 응답하지 않아 멈췄습니다", "모터 응답 없음"],
    motor: ["모터가 응답하지 않아 시작하지 못했습니다", "모터 응답 없음"],
    nopolicy: ["키트에 정책이 없습니다", "키트 오류"],
    tilt: ["시작 못 함 (받침이 너무 기울었습니다. 수평을 맞춘 뒤 다시 하세요)", "받침 기울어짐"],
    level: ["시작 못 함 (진자 영점이 맞지 않습니다. 키트 맞추기를 다시 하세요)", "영점 다름"],
    error: ["키트 오류로 멈췄습니다", "키트 오류"]
  };
  var KITMSG = {
    upload: "정책을 키트로 보내는 중…",
    check: "키트 점검 중…",
    still: "진자가 멈추기를 기다리는 중…",
    brake: "브레이크 푸는 중…",
    level: "받침 수평 재는 중…",
    start: ""
  };

  /* 실행 전 안내 한 줄. 수평 측정은 "level 1/3 …" 다음 "level ok tilt=… dir=…" 로 옵니다. */
  function kitMsgText(code, args) {
    if (code === "still" && args[0]) return KITMSG.still + " " + args[0] + "초";
    if (code === "level" && args[0] === "ok") {
      var t = +((args[1] || "").split("=")[1] || 0);
      return "받침 기울기 " + t.toFixed(1) + "° → 이번 실행의 진자 영점을 고쳤습니다";
    }
    if (code === "level" && (args[0] || "").indexOf("/") >= 0) return "받침 수평 재는 중 (" + args[0] + ")…";
    return KITMSG[code] || "";
  }

  Judge.prototype.outcome = function () {
    if (this.end && this.end.reason !== "done") {
      var k = KITEND[this.end.reason] || KITEND.error;
      return { long: k[0], short: k[1], ok: false };
    }
    if (this.stopped) return { long: "안전 정지 (너무 빨리 돌았습니다)", short: "안전 정지", ok: false };
    if (this.up === null) return { long: "못 세움", short: "못 세움", ok: false };
    var t = this.up.toFixed(1) + "초";
    if (this.fell !== null) return { long: t + " 만에 세움 → 쓰러짐", short: t + " · 쓰러짐", ok: false };
    var shake = this.wob > 3;
    return {
      long: t + " 만에 세움 · 끝까지 버팀" + (shake ? " (덜덜 떨림)" : ""),
      short: t + " · 버팀" + (shake ? "(떨림)" : ""), ok: true
    };
  };
  Judge.prototype.metrics = function () {
    return {
      up: this.up,
      force: this.fn ? this.fsum / this.fn * 100 : null,
      arm: this.th0 === null ? null : Math.abs(this.th1 - this.th0) / (2 * Math.PI)
    };
  };

  /* ── 가치: 그 순간부터 앞으로 받을 보상을 가까운 것일수록 크게 쳐서 더한 값.
   *    시뮬레이션에서 같은 정책을 미리 한 번 돌려 구합니다. ── */
  function valueTrace(mlp, r, secs) {
    var sim = new PS.Furuta(C), obs = new Float64Array(PS.OBS_N);
    var n = Math.round(secs / sim.ctrlDt), R = new Float64Array(n), G = new Float64Array(n), g = 0.99;
    sim.reset(0, 0, 0, 0);
    for (var i = 0; i < n; i++) {
      sim.obs(obs);
      var u = mlp.forward(obs);
      sim.step(u);
      R[i] = r.fn(sim.phi(), sim.ald, sim.thd, u, sim.th);
    }
    G[n - 1] = R[n - 1] / (1 - g);
    for (i = n - 2; i >= 0; i--) G[i] = R[i] + g * G[i + 1];
    var lo = Infinity, hi = -Infinity;
    for (i = 0; i < n; i++) { lo = Math.min(lo, G[i]); hi = Math.max(hi, G[i]); }
    var span = hi - lo || 1;
    for (i = 0; i < n; i++) G[i] = (G[i] - lo) / span;
    return G;
  }

  /* ── 한 번 시험하기 ── */
  function runTest(dev, label, r, mlp, secs, opts) {
    opts = opts || {};
    var judge = new Judge(), vals = valueTrace(mlp, r, secs);
    var rec = opts.record ? st.runData : null;
    stage.label(label);
    stage.busy(true);
    stage.clearResult();
    var dt = dev.ctrlDt, lastT = 0, n = 0;
    if (dev.kind === "kit") stage.show({ t: 0, th: 0, al: 0, thd: 0, ald: 0, u: 0, flags: 0 }, { reward: 0, value: 0, peak: 0, tend: secs });
    return PS.runEpisode(dev, function (o) { return mlp.forward(o); }, {
      seconds: secs,
      policy: mlp,
      onMsg: function (code, args) {
        stage.note(kitMsgText(code, args));
        if (code === "start") say("키트 실행 중");
      },
      onTick: function (s) {
        var phi = PS.wrap(s.al - Math.PI);
        judge.tick(s, phi, Math.max(dt, s.t - lastT));
        lastT = s.t;
        var rv = r.fn(phi, s.ald, s.thd, s.u, s.th);
        var vi = Math.min(vals.length - 1, Math.max(0, Math.round(s.t / dt)));
        stage.show(s, { reward: rv, value: vals[vi], peak: judge.peak, tend: secs });
        n++;
        if (opts.live && r.trace && n % 3 === 0) blocks.live(r.trace(phi, s.ald, s.thd, s.u, s.th));
        if (rec) {
          rec.t.push(s.t);
          rec.height.push(50 * (1 + Math.cos(phi)));
          rec.force.push(Math.abs(s.u) * 100);
          rec.arm.push(Math.abs(s.th - judge.th0) / (2 * Math.PI));
          if (n % 6 === 0 && st.tab === "graphs") drawCharts("test");
        }
      },
      onDone: function (s, end) {
        if (end) {
          judge.end = end;
          stage.note("");
          say("");
          id("alertLine").textContent = end.reason === "done" || end.reason === "stop" || !end.detail ? "" : "키트: " + end.detail;
        }
        var out = judge.outcome();
        stage.busy(false);
        stage.show(s, { reward: 0, value: 0, peak: judge.peak, tend: secs, result: "결과: " + out.long });
        blocks.live(null);
        if (rec) drawCharts("test");
        if (opts.onDone) opts.onDone(out, judge);
      }
    });
  }

  /* ── 살펴보기: 미리 배운 에이전트 ── */
  id("runPre").addEventListener("click", function () {
    if (st.busy) return;
    var r = PS.reward.basic(1, 1), got = PS.pickPolicy(r);
    if (!got) { alert("번들된 정책이 없습니다. rl/export_for_app.py 를 먼저 실행하세요."); return; }
    setBusy(true);
    id("preResult").hidden = true;
    st.run = runTest(simDev, "시뮬레이션 (미리 배운 에이전트)", r, new PS.MLP(got.spec), 10, {
      onDone: function (out) {
        id("preResult").textContent = out.long;
        id("preResult").hidden = false;
        st.done1 = true;
        st.run = null;
        setBusy(false);
      }
    });
  });
  id("toBlocks").addEventListener("click", function () { setStep(2); });

  /* ── 규칙이 바뀌었을 때 ── */
  function changed() {
    if (st.busy) return;
    st.policy = null;
    st.policyKey = null;
    st.simOK = false;
    st.gate = null;
    st.fallback = null;
    id("wrote").checked = false;
    id("trainBar").style.width = "0";
    id("learnState").textContent = "모델없음";
    id("trainMsg").textContent = "시작하기를 누르면 배웁니다";
    id("resBanner").hidden = true;
    say("");
    refresh();
  }

  /* ── 배우기 ── */
  function drawCurve(points) {
    st.runData.curve = points.slice();
    if (st.tab === "graphs") drawCharts("learn");
  }

  function learn(r, o, key, then) {
    var got = PS.pickPolicy(r);
    if (!got) { alert("번들된 정책이 없습니다."); setBusy(false); return; }
    var base = new PS.MLP(got.spec);
    id("learnState").textContent = "학습중";
    blocks.mark("learn");

    if (r.kind === "basic" && got.exact) {
      /* 미리 학습해 둔 결과를 불러옵니다. 그때 실제로 기록된 점수 곡선을 그대로 보여 줍니다. */
      id("trainMsg").textContent = "미리 학습해 둔 정책을 불러옵니다";
      var curve = got.spec.curve || [], shown = [], i = 0, t0 = performance.now();
      st.anim = { stop: false };
      var anim = st.anim;
      (function tick() {
        if (anim.stop) return;
        var p = Math.min(1, (performance.now() - t0) / 2200);
        id("trainBar").style.width = (p * 100) + "%";
        var want = Math.round(p * curve.length);
        while (i < want) shown.push(curve[i++]);
        drawCurve(shown);
        if (p < 1) { setTimeout(tick, 30); return; }
        st.anim = null;
        st.policy = base;
        st.policyKey = key;
        st.gate = got.spec.gate === undefined ? null : got.spec.gate;
        st.fallback = null;
        id("learnState").textContent = "학습완료";
        id("trainMsg").textContent = "미리 학습해 둔 정책" + (st.gate === null ? "" : " · 안전 시험 " + st.gate + "/" + GT);
        then();
      })();
      return;
    }

    /* 새로 만든 규칙: 미리 배운 정책에서 출발해 지금 고쳐 배우고, 안전 시험으로 고릅니다. */
    var started = performance.now();
    st.trainer = PS.learn.run(C, base, r,
      { gens: o.gens, pop: o.pop, elite: Math.max(3, Math.round(o.pop / 4)), sigma: o.sigma },
      function (p, hist, tr, phase) {
        id("trainBar").style.width = (p * 100) + "%";
        id("trainMsg").textContent = phase === "gate" ? "안전 시험을 보는 중" : "세대 " + tr.gen + " / " + tr.gens;
        drawCurve(hist);
      },
      function (res) {
        st.trainer = null;
        st.policy = res.mlp;
        st.policyKey = key;
        st.gate = res.gate;
        st.fallback = res.fallback;
        st.learnedSec = (performance.now() - started) / 1000;
        id("learnState").textContent = "학습완료";
        id("trainMsg").textContent = res.fallback === "gate"
          ? "고친 정책이 안전 시험에서 떨어졌습니다 (" + res.gate + "/" + GT + "). 미리 배운 정책으로 시험합니다"
          : res.fallback === "noimprove"
            ? "고쳐도 점수가 오르지 않아 미리 배운 정책을 그대로 씁니다"
            : st.learnedSec.toFixed(0) + "초 동안 고쳐 배웠습니다 · 안전 시험 " + res.gate + "/" + GT;
        then();
      });
  }

  /* ── 결과 카드와 기록 ── */
  function fmtPrev(now, prev, digits, unit) {
    var s = now.toFixed(digits) + unit;
    if (prev !== null && prev !== undefined) s += " <small>(비교 " + prev.toFixed(digits) + ")</small>";
    return s;
  }
  function paintResult() {
    var L = st.lastResult;
    if (!L) return;
    var m = L.m, b = id("resBanner"), cr = compareRow(), p = cr && cr.m ? cr.m : {};
    var min = C.gate.pass_min, good = L.pass;
    b.hidden = false;
    b.className = "banner " + (good ? "ok" : "bad");
    if (L.kit) b.textContent = L.ok ? "✔ 실제 키트에서 세웠어요" : "✖ 실제 키트에서 못 세웠어요 · 시뮬레이션과 달랐던 점을 적어 보세요";
    else if (L.fallback === "gate") b.textContent = "✖ 이 규칙으로 고친 정책은 안전 시험에서 떨어졌어요 (" + L.gate + "/" + GT + ") · 규칙을 고쳐 보세요";
    else if (L.gate !== null && L.gate < min) b.textContent = "✖ 안전 시험 " + L.gate + "/" + GT + " · 실제 키트에는 올릴 수 없어요";
    else if (!L.ok) b.textContent = "✖ 시뮬레이션에서 못 세웠어요 · 점수 규칙을 고쳐 보세요";
    else b.textContent = "✔ 안전 시험 통과" + (L.gate === null ? "" : " (" + L.gate + "/" + GT + ")") + " · 실제 키트에서 시험할 수 있어요";
    id("mUp").innerHTML = m.up === null ? "못 세움" : fmtPrev(m.up, p.up, 1, "초");
    id("mForce").innerHTML = m.force === null ? "-" : fmtPrev(m.force, p.force, 0, "");
    id("mArm").innerHTML = m.arm === null ? "-" : fmtPrev(m.arm, p.arm, 1, "바퀴");
  }
  function paintRows() {
    var cr = compareRow();
    st.rows.forEach(function (row) {
      var c = row.tr.cells, m = row.m || {};
      row.tr.className = "row" + (row === st.curRow ? " cur" : "") + (row === cr ? " cmp" : "");
      c[0].innerHTML = row.no + "회" + (row === st.curRow ? '<span class="tag cur">이번</span>' : row === cr ? '<span class="tag cmp">비교</span>' : "");
      c[1].textContent = row.label;
      c[1].title = row.text;
      c[2].textContent = row.sim;
      c[3].textContent = row.kit;
      c[4].textContent = m.force === null || m.force === undefined ? "-" : m.force.toFixed(0);
      c[5].textContent = m.arm === null || m.arm === undefined ? "-" : m.arm.toFixed(1);
    });
  }
  function logRow(r, kit, out, judge) {
    var log = id("log");
    if (!st.curRow) {
      if (!st.rows.length) log.innerHTML = "";
      st.runNo++;
      var row = { no: st.runNo, label: r.label || "직접 만든 규칙", text: r.text || "", sim: "-", kit: "-", m: null, data: null, tr: log.insertRow(0) };
      row.tr.innerHTML = '<td></td><td></td><td></td><td></td><td class="num"></td><td class="num"></td>';
      row.tr.addEventListener("click", function () {
        if (row === st.curRow) return;
        st.pin = st.pin === row ? null : row;
        paintRows();
        paintResult();
        drawCharts();
      });
      st.rows.unshift(row);
      st.curRow = row;
      while (st.rows.length > 8) {
        var old = st.rows.pop();
        if (st.pin === old) st.pin = null;
        old.tr.remove();
      }
    }
    st.curRow[kit ? "kit" : "sim"] = out.short;
    st.curRow.m = judge.metrics();
    st.curRow.data = st.runData;
    var gateOK = st.fallback !== "gate" && (st.gate === null || st.gate >= C.gate.pass_min);
    st.lastResult = { m: st.curRow.m, ok: out.ok, kit: kit, gate: st.gate, fallback: st.fallback, pass: kit ? out.ok : out.ok && gateOK };
    paintRows();
    paintResult();
  }

  function test(r, secs) {
    var kit = st.env === "kit", dev = kit ? st.kit : simDev;
    id("learnState").textContent = "학습완료";
    blocks.mark("test");
    st.run = runTest(dev, kit ? "실제 키트" : "시뮬레이션", r, st.policy, secs, {
      record: true, live: true,
      onDone: function (out, judge) {
        st.run = null;
        logRow(r, kit, out, judge);
        drawCharts();
        if (!kit) st.simOK = st.lastResult.pass;
        blocks.mark(null);
        setBusy(false);
      }
    });
  }

  /* ── 시작하기 / 정지하기 ── */
  function start() {
    if (st.busy) return;
    var r = rule(), o = runOpts(), key = keyOf(r, o);
    if (!r.ok) { say("점수 블록이 없습니다. 점수 범주에서 끌어다 놓으세요."); setTab("blocks"); return; }
    if (!id("wrote").checked) {
      say("학습지에 예상을 먼저 적고, 위의 '학습지에 예상을 적었어요'에 표시하세요.");
      id("wroteBox").classList.add("need");
      return;
    }
    var need = !st.policy || st.policyKey !== key;
    if (st.env === "kit") {
      if (!st.kitReady) { openDevices(); return; }
      if (st.kit.running) { say("키트가 아직 멈추는 중입니다. 잠깐 뒤에 다시 누르세요."); return; }
      if (need || !st.simOK) { say("먼저 시뮬레이션에서 안전 시험을 통과해야 키트에서 시험할 수 있어요."); return; }
    }
    if (need && !o.learn) { say("'내 점수 규칙으로 정책 학습하기' 블록을 시작하기 아래에 붙이세요."); return; }
    if (!need && !o.test) { say("이미 배웠습니다. '초 동안 시험하기' 블록을 붙이면 시험합니다."); return; }
    say("");
    setStep(3, true);
    setBusy(true);
    /* 새로 배우면 기록 표에 새 줄이 생깁니다. 같은 정책을 다시 시험만 하면 같은 줄을 고쳐 씁니다. */
    var keep = need ? [] : st.runData.curve;
    if (need) st.curRow = null;
    st.runData = newRun();
    st.runData.curve = keep;
    paintRows();
    st.runData.label = (r.label || "") + (st.env === "kit" ? " · 키트" : "");
    st.runData.secs = o.test || 10;
    drawCharts();
    var after = function () {
      if (o.test) test(r, o.test);
      else { blocks.mark(null); say("배우기만 했습니다. '초 동안 시험하기' 블록을 붙이면 시험합니다."); setBusy(false); }
    };
    if (need) learn(r, o, key, after); else after();
  }
  function stop() {
    if (!st.busy) return;
    if (st.run) { st.run.cancel(); st.run = null; }
    if (st.trainer) { st.trainer.stopped = true; st.trainer = null; id("learnState").textContent = "모델없음"; id("trainBar").style.width = "0"; }
    if (st.anim) { st.anim.stop = true; st.anim = null; id("learnState").textContent = "모델없음"; id("trainBar").style.width = "0"; }
    stage.busy(false);
    stage.result("정지했습니다");
    blocks.live(null);
    blocks.mark(null);
    say("정지했습니다.");
    setBusy(false);
  }
  id("btnStart").addEventListener("click", start);
  id("btnStop").addEventListener("click", stop);
  id("wrote").addEventListener("change", function () { id("wroteBox").classList.remove("need"); say(""); refresh(); });

  /* ── 따라 하기 / 자유롭게 만들기 ── */
  function setMode(m) {
    st.mode = m;
    blocks.setMode(m, { smooth: st.smooth, effort: st.effort });
    changed();
  }
  id("modeSeg").querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () { if (!st.busy) { setMode(b.dataset.m); setStep(2); } });
  });

  /* ── 시뮬레이션 / 실제 키트 ── */
  id("envSeg").querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () {
      if (st.busy) return;
      st.env = b.dataset.env;
      if (st.env === "kit" && !st.kitReady) openDevices();
      say("");
      refresh();
    });
  });

  /* ── 탭 ── */
  function setTab(tab) {
    st.tab = tab;
    document.querySelectorAll(".mainpane").forEach(function (p) { p.classList.toggle("on", p.dataset.pane === tab); });
    id("mainTabs").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.tab === tab)); });
    if (tab === "blocks") blocks.refresh();
    if (tab === "graphs") drawCharts();
  }
  id("mainTabs").querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () {
      setTab(b.dataset.tab);
      if (b.dataset.tab === "intro") setStep(1, true);
      else if (st.step === 1) setStep(2, true);
    });
  });
  id("subTabs").querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () {
      id("subTabs").querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
      document.querySelectorAll(".subpane").forEach(function (p) { p.classList.toggle("on", p.dataset.pane === b.dataset.sub); });
    });
  });

  /* ── 활동 순서 ── */
  function setStep(n, keepTab) {
    st.step = n;
    if (!keepTab) setTab(n === 1 ? "intro" : "blocks");
    refresh();
  }
  document.querySelectorAll(".gstep").forEach(function (b) {
    b.addEventListener("click", function () { setStep(+b.dataset.step); });
  });

  /* ── 예시 규칙 ── */
  var EXAMPLES = PS.EXAMPLES;
  function paintExamples() {
    var box = id("examples");
    box.innerHTML = "";
    EXAMPLES.forEach(function (ex) {
      var c = document.createElement("section");
      c.className = "card excard" + (ex.bad ? " bad" : "");
      c.innerHTML = "<b></b><p class='muted'></p><button class='big alt'>이 규칙 불러오기</button>";
      c.querySelector("b").textContent = ex.name;
      c.querySelector("p").textContent = ex.why;
      c.querySelector("button").addEventListener("click", function () {
        if (st.busy) return;
        if (st.mode !== "free") { st.mode = "free"; blocks.setMode("free"); }
        blocks.loadReward(ex.nodes);
        setStep(2);
      });
      box.appendChild(c);
    });
  }

  /* ── 실행 화면 크게 보기 ── */
  $(".bigbtn").addEventListener("click", function () {
    var on = $(".body").classList.toggle("stagebig");
    $(".bigbtn").setAttribute("aria-pressed", String(on));
    $(".bigbtn span").textContent = on ? "작게 보기" : "크게 보기";
    if (!on) { if (st.tab === "blocks") blocks.refresh(); drawCharts(); }
  });

  /* ── 되돌리기 / 다시 실행 ── */
  id("mUndo").addEventListener("click", function () { if (!st.busy) blocks.undo(); });
  id("mRedo").addEventListener("click", function () { if (!st.busy) blocks.redo(); });
  window.addEventListener("keydown", function (e) {
    if (st.busy || !(e.ctrlKey || e.metaKey)) return;
    var tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    var k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) { blocks.undo(); e.preventDefault(); }
    else if (k === "y" || (k === "z" && e.shiftKey)) { blocks.redo(); e.preventDefault(); }
  });

  /* ── 새로 만들기 / 불러오기 / 저장하기 / 도움말 ── */
  id("mNew").addEventListener("click", function () {
    if (st.busy) return;
    if (!confirm("블록을 처음 상태로 되돌릴까요?")) return;
    st.smooth = 1; st.effort = 1;
    blocks.reset();
    setMode(st.mode);
  });
  id("mSave").addEventListener("click", function () {
    var data = { app: "pendulum-studio", v: 2, name: id("projName").value, mode: st.mode, smooth: st.smooth, effort: st.effort, blocks: blocks.exportModel() };
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
    a.download = (id("projName").value || "작품") + ".json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  });
  id("mOpen").addEventListener("click", function () { if (!st.busy) id("fileIn").click(); });
  id("fileIn").addEventListener("change", function () {
    var f = id("fileIn").files[0];
    if (!f) return;
    var rd = new FileReader();
    rd.onload = function () {
      try {
        var d = JSON.parse(rd.result);
        if (d.app !== "pendulum-studio" || !blocks.importModel(d.blocks)) throw new Error("형식");
        st.smooth = d.smooth === undefined ? 1 : +d.smooth;
        st.effort = d.effort === undefined ? 1 : +d.effort;
        if (d.name) id("projName").value = d.name;
        setMode(d.mode === "free" ? "free" : "guide");
        setStep(2);
      } catch (e) {
        alert("이 파일은 불러올 수 없습니다. Pendulum Studio 에서 저장한 파일인지 확인하세요.");
      }
      id("fileIn").value = "";
    };
    rd.readAsText(f);
  });
  id("mHelp").addEventListener("click", function () {
    alert("활동 순서\n" +
      "1 살펴보기: 미리 배운 에이전트를 실행해 봅니다 (학습지 2쪽).\n" +
      "2 점수 규칙 만들기: '매 순간 점수 매기기' 아래에 블록을 쌓습니다 (학습지 3쪽).\n" +
      "3 가르치고 시험하기: 시작하기를 누르면 배우고 나서 시험합니다.\n\n" +
      "블록을 누르면 왼쪽 아래 '블록 도움말'에 설명이 나옵니다.\n" +
      "블록을 버리려면 휴지통이나 블록 꾸러미로 끌어다 놓으세요.");
  });

  /* ── 키트 연결 ── */
  var devSel = null, devPorts = [];

  function openDevices() {
    id("devModal").classList.add("on");
    refreshPorts();
  }
  function closeDevices() { id("devModal").classList.remove("on"); }

  function refreshPorts() {
    id("devNote").textContent = "키트를 찾는 중…";
    var api = window.studio && window.studio.serial;
    if (!api) {
      devPorts = [];
      id("devNote").textContent = "이 창에서는 키트를 연결할 수 없습니다. 설치한 Pendulum Studio 앱에서 연결하세요.";
      paintDevices();
      return;
    }
    api.available().then(function (ok) {
      if (!ok) {
        devPorts = [];
        id("devNote").textContent = "serialport 모듈이 설치되어 있지 않습니다. npm install 후 다시 열어 주세요.";
        paintDevices();
        return;
      }
      return api.list().then(function (ps) {
        devPorts = ps;
        id("devNote").textContent = ps.length ? "키트를 고르고 연결을 누르세요." : "키트를 찾지 못했습니다. USB 케이블을 확인하세요.";
        paintDevices();
      });
    }).catch(function (e) {
      id("devNote").textContent = String(e && e.message || e);
      paintDevices();
    });
  }

  function paintDevices() {
    var grid = id("devGrid");
    grid.innerHTML = "";
    devPorts.forEach(function (p) {
      var b = document.createElement("button");
      b.className = "devcard";
      b.setAttribute("aria-pressed", String(devSel === p.path));
      b.innerHTML = '<span class="ic">🔌</span><b></b><span></span>';
      b.querySelector("b").textContent = p.path;
      b.querySelector("span:last-child").textContent = (p.manufacturer || "USB 직렬 장치") + (p.vendorId ? "  " + p.vendorId + ":" + p.productId : "");
      b.addEventListener("click", function () { devSel = p.path; paintDevices(); });
      grid.appendChild(b);
    });
    id("devConnect").disabled = !devSel;
  }

  id("mConn").addEventListener("click", openDevices);
  id("kitConnect").addEventListener("click", openDevices);
  id("devClose").addEventListener("click", closeDevices);
  id("devRefresh").addEventListener("click", refreshPorts);
  id("devConnect").addEventListener("click", function () {
    if (!devSel) return;
    var dev = new PS.KitDevice(devSel);
    id("devNote").textContent = "연결하는 중…";
    id("devConnect").disabled = true;
    var old = st.kit && st.kit.connected ? st.kit.disconnect() : Promise.resolve();
    old.then(function () { return dev.connect(); }).then(function () {
      st.kit = dev;
      setConn(true, "키트 연결됨 · " + dev.name + (dev.info.fake ? " (가짜 장치)" : ""));
      id("kitNote").textContent = "키트 프로그램 " + dev.info.fw + " · 제어 " + dev.info.hz + " Hz · 모터 힘 한계 " +
        (dev.info.tau_mNm / 1000).toFixed(2) + " N·m. 정책은 키트 안에서 실행됩니다.";
      id("kitCheckOut").textContent = "";
      dev.on("alert", function (a) { id("alertLine").textContent = "키트: " + a.code + " " + a.message; });
      dev.on("status", function (d) { if (!d.connected) { setConn(false, "연결이 끊어졌습니다"); refresh(); } });
      closeDevices();
      refresh();
      dev.calInfo().then(function () { paintCalNote(); openCal(true); }).catch(function () { openCal(true); });
    }).catch(function (e) {
      id("devConnect").disabled = false;
      id("devNote").textContent = "연결 실패: " + (e && e.message || e);
    });
  });

  /* 키트 점검: 모터를 움직이지 않고 읽기만 합니다. */
  id("kitCheck").addEventListener("click", function () {
    var out = id("kitCheckOut");
    if (!st.kitReady || st.busy || st.kit.running) { out.textContent = "키트를 연결하고, 실행이 끝난 뒤에 눌러 주세요."; return; }
    out.textContent = "점검하는 중…";
    st.kit.check().then(function (c) {
      if (!c.ok) { out.textContent = "✖ 점검에서 걸렸습니다: " + c.text; return; }
      var off = (c.raw - c.zero) * 360 / 4096, hang = Math.abs(off) <= 5.3;
      out.textContent = "✔ 모터 응답 정상 · 모터 온도 " + c.temp + "°C · 진자 " +
        (hang ? "아래로 매달림 (영점에서 " + off.toFixed(1) + "°)" : "아래에서 " + off.toFixed(0) + "° 벗어남 (멈출 때까지 기다린 뒤 시작합니다)");
    }).catch(function (e) { out.textContent = "✖ " + (e && e.message || e); });
  });

  /* ── 키트 맞추기 (연결할 때). 모터가 움직이므로 확인 단추를 눌러야 시작합니다.
   *    학습 범위는 s2r/s2r_env.py 의 RANGE (정책을 배운 조건의 폭) 그대로입니다. ── */
  var TRAIN = { Jm: [0.0210, 0.0260, 0.0245], Fc: [0.42, 0.85, 0.69], b: [0.06, 0.30, 0.17] };

  function stampText(s) {
    if (!s) return "";
    var m = /^(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)/.exec(s);
    return m ? m[1] + "-" + m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] : s;
  }

  function paintCalNote() {
    var c = st.kit && st.kit.cal, el = id("kitCalOut");
    if (!c) { el.textContent = ""; return; }
    el.textContent = c.none ? "키트 맞추기를 아직 하지 않았습니다. 진자 영점은 기본값을 씁니다." :
      "마지막 키트 맞추기 " + stampText(c.stamp) + " (키트 시계) · 팔 관성 " + (c.Jm || 0).toFixed(4) +
      " · 진자 주기 " + (c.T0 ? c.T0.toFixed(3) + " s" : "-") + (c.ok ? "" : " · 다시 맞추기를 권합니다");
  }

  function calRow(name, val, range, unit, digits) {
    var ok = val !== null && val !== undefined && val >= range[0] && val <= range[1];
    return "<tr><td>" + name + "</td><td><b>" + (val === null || val === undefined ? "-" : val.toFixed(digits)) + "</b> " + unit +
      "</td><td>" + range[0] + " ~ " + range[1] + " (공칭 " + range[2] + ")</td><td class=\"" + (ok ? "ok" : "warn") + "\">" +
      (ok ? "✔ 범위 안" : "⚠ 범위 밖") + "</td></tr>";
  }

  function paintCalResult(c) {
    var d = c.T0 && c.T0m ? (c.T0 / c.T0m - 1) * 100 : null, pOk = d !== null && Math.abs(d) <= 2;
    id("calTable").innerHTML = "<tr><th>항목</th><th>이 키트</th><th>정책이 배운 폭</th><th></th></tr>" +
      calRow("팔 관성 (모터)", c.Jm, TRAIN.Jm, "kg·m²", 4) +
      calRow("팔 마찰 + 방향", c.Fp, TRAIN.Fc, "N·m", 2) +
      calRow("팔 마찰 − 방향", c.Fm, TRAIN.Fc, "N·m", 2) +
      calRow("팔 점성", c.b, TRAIN.b, "N·m·s", 3) +
      "<tr><td>진자 흔들림 주기</td><td><b>" + (c.T0 ? c.T0.toFixed(3) : "-") + "</b> s</td><td>모델 " + (c.T0m || 0).toFixed(3) +
      " s</td><td class=\"" + (pOk ? "ok" : "warn") + "\">" + (d === null ? "⚠ 못 잼" : (pOk ? "✔ " : "⚠ ") + "모델과 " +
      (d >= 0 ? "+" : "") + d.toFixed(1) + "%") + "</td></tr>" +
      "<tr><td>받침 기울기</td><td><b>" + (c.tilt || 0).toFixed(1) + "</b>°</td><td>실행마다 다시 잽니다</td><td class=\"" +
      ((c.tilt || 0) <= 1 ? "ok" : "warn") + "\">" + ((c.tilt || 0) <= 1 ? "✔ 평평함" : "⚠ 수평을 맞추면 좋습니다") + "</td></tr>";
    id("calTable").hidden = false;
  }

  function openCal(fromConnect) {
    if (!st.kitReady) return;
    var c = st.kit.cal;
    id("calLast").textContent = !c ? "" : c.none ? "이 키트는 아직 맞춘 적이 없습니다." :
      "마지막으로 맞춘 때: " + stampText(c.stamp) + " (키트 시계). 그 값을 그대로 써도 됩니다.";
    id("calProg").textContent = "";
    id("calTable").hidden = true;
    id("calGo").disabled = false;
    id("calGo").textContent = "맞추기 시작";
    id("calLater").textContent = fromConnect ? "나중에" : "닫기";
    if (c && !c.none) paintCalResult(c);
    id("calModal").classList.add("on");
  }

  id("calLater").addEventListener("click", function () {
    if (st.kit && st.kit._cal) { st.kit.stop(); return; }
    id("calModal").classList.remove("on");
  });
  id("kitCal").addEventListener("click", function () { openCal(false); });
  id("calGo").addEventListener("click", function () {
    var dev = st.kit;
    if (!dev || !st.kitReady || st.busy || dev.running) { id("calProg").textContent = "실행이 끝난 뒤에 눌러 주세요."; return; }
    var steps = { check: "키트 점검 중…", still: "진자가 멈추기를 기다리는 중…", brake: "브레이크 푸는 중…", arm: "팔 마찰·관성 재는 중 (팔이 세게 돌았다 멈춥니다)…",
      swing: "팔을 잠그고 진자 흔들림 재는 중 (10초)…" };
    id("calGo").disabled = true;
    id("calLater").textContent = "멈추기";
    id("calTable").hidden = true;
    id("calProg").textContent = "시작합니다…";
    setBusy(true);
    dev.calibrate({
      onMsg: function (code, args) {
        if (code === "level" && (args[0] || "").indexOf("/") >= 0) id("calProg").textContent = "진자 영점 재는 중 (" + args[0] + ", 팔이 90°씩 돕니다)…";
        else if (code === "level" && args[0] === "ok") id("calProg").textContent = "진자 영점 끝 · " + args.slice(1).join(" ");
        else if (steps[code] && args[0] !== "ok") id("calProg").textContent = steps[code];
      }
    }).then(function (c) {
      id("calProg").textContent = c.ok ? "✔ 다 맞췄습니다. 키트에 저장했고, 다음 실행부터 이 영점으로 수평을 잽니다." :
        "⚠ 끝났지만 일부 값이 이상합니다. 키트를 확인하고 다시 맞춰 보세요.";
      id("calLast").textContent = "방금 맞췄습니다: " + stampText(c.stamp) + " (키트 시계)";
      paintCalResult(c);
      paintCalNote();
    }).catch(function (e) {
      id("calProg").textContent = "✖ 키트 맞추기를 끝내지 못했습니다: " + (e && e.message || e);
    }).then(function () {
      id("calGo").disabled = false;
      id("calGo").textContent = "다시 맞추기";
      id("calLater").textContent = "닫기";
      setBusy(false);
    });
  });

  function setConn(on, text) {
    st.kitReady = on;
    id("kitCheck").disabled = !on;
    id("kitCal").disabled = !on;
    if (!on) { id("kitNote").textContent = "USB 케이블로 키트를 연결한 뒤 아래 버튼을 누르세요."; id("kitCalOut").textContent = ""; if (st.env === "kit") st.env = "sim"; }
    id("connName").textContent = on ? text : "키트 연결하기";
    id("mConn").classList.toggle("on", on);
    id("kitDot").classList.toggle("on", on);
    id("kitState").textContent = on ? text : (text || "키트가 연결되지 않았습니다");
  }

  /* ── 새로 그리기 ── */
  function setBusy(on) {
    st.busy = on;
    refresh();
  }
  function refresh() {
    var r = rule(), lock = st.busy;
    blocks.setCheck(PS.reward.check(r));
    if (lock !== st.locked) { st.locked = lock; blocks.lock(lock); }
    id("btnStart").disabled = lock;
    id("btnStop").disabled = !lock;
    id("wrote").disabled = lock;
    id("runPre").disabled = lock;
    id("mUndo").disabled = lock || !blocks.canUndo();
    id("mRedo").disabled = lock || !blocks.canRedo();
    id("modeSeg").querySelectorAll("button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.m === st.mode));
      b.disabled = lock;
    });
    id("envSeg").querySelectorAll("button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.env === st.env));
      b.disabled = lock;
    });
    document.querySelectorAll(".gstep").forEach(function (b) {
      var n = +b.dataset.step;
      b.setAttribute("aria-current", String(n === st.step));
      b.classList.toggle("done", (n === 1 && st.done1) || (n === 2 && id("wrote").checked) || (n === 3 && !!st.simOK));
    });
    id("guideHint").textContent = HINT[st.step]();
    id("wroteBox").style.visibility = st.step === 1 ? "hidden" : "visible";
  }

  /* ── 시작 ── */
  function boot() {
    if (!C) {
      document.body.innerHTML = '<p style="padding:40px;font-size:18px">모델 파일이 없습니다. ' +
        '프로젝트 루트에서 <code>python rl/export_for_app.py</code> 를 먼저 실행하세요.</p>';
      return;
    }
    id("modelNote").textContent = "rev11 실기 보정 모델 · 제어 " + C.ctrl_hz + " Hz · 미리 배운 정책 " +
      Object.keys(PS.POLICIES || {}).length + "개";
    simDev.reset(0, 0, 0, 0);
    stage.show(simDev.state(), { tend: 10 });
    setConn(false);
    id("trainMsg").textContent = "시작하기를 누르면 배웁니다";
    paintExamples();
    drawCharts();
    blocks.setMode("guide", { smooth: st.smooth, effort: st.effort });
    setStep(1);
    /* 주소 뒤에 ?mode=free 또는 ?step=2 를 붙이면 그 상태로 엽니다 (미리보기·수업 자료용). */
    var q = new URLSearchParams(location.search);
    if (q.get("mode") === "free") setMode("free");
    if (q.get("step")) setStep(Math.max(1, Math.min(3, +q.get("step") || 1)));
  }

  boot();
})(window.PS = window.PS || {});
