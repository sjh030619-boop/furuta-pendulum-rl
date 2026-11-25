/* 점수 규칙(보상).
 *
 * 보상 함수 서명은 어디서나 같습니다.
 *   fn(phi, alphaDot, thetaDot, u, theta) -> 점수        (phi = 똑바로 섰을 때 0)
 *
 * 따라 하기: 학습 때 쓴 환경(rl/train_rules.py)의 보상과 같은 식이고, 두 고르는 칸이 가중치를 정합니다.
 *   r = 1 − 0.3 φ² − w_smooth·α̇² − 0.002·θ̇² − w_effort·u² + 0.5·exp(−(φ/0.3)²)·exp(−(α̇/2)²)
 * 자유롭게 만들기: 학생이 쌓은 블록을 그대로 계산합니다.
 *
 * 블록 모델 (blocks.js 가 만듦)
 *   B[id] = { id, type, s: [{ block, v }], f: { 고르는 칸 }, bodies: [[id...], ...] }
 *   scripts.reward = [id...]   매 순간 점수 매기기 아래
 *   scripts.run    = [id...]   시작하기 버튼을 클릭했을 때 아래
 */
(function (PS) {
  "use strict";

  var WORD = ["없음", "약하게", "강하게"];
  var W_SMOOTH = [0, 0.002, 0.008];
  var W_EFFORT = [0, 0.02, 0.10];

  function basic(smooth, effort) {
    var ws = W_SMOOTH[smooth], we = W_EFFORT[effort];
    var text = "점수 = 세워짐";
    if (smooth) text += " − 빠르기(" + WORD[smooth] + ")";
    if (effort) text += " − 힘(" + WORD[effort] + ")";
    function fn(phi, ald, thd, u) {
      var near = Math.exp(-11.111111 * phi * phi);
      var slow = Math.exp(-0.25 * ald * ald);
      return 1 - 0.3 * phi * phi - ws * ald * ald - 0.002 * thd * thd
        - we * u * u + 0.5 * near * slow;
    }
    return {
      kind: "basic", smooth: smooth, effort: effort, ok: true,
      text: text, label: "빠르기 " + WORD[smooth] + ", 힘 " + WORD[effort],
      key: "s" + smooth + "e" + effort, fn: fn
    };
  }

  /* ── 블록 계산 ── */
  function numOf(M, slot, v, depth) {
    return slot.block ? valOf(M, slot.block, v, depth) : slot.v;
  }
  function boolSlot(M, slot, v, depth) {
    return slot.block ? boolOf(M, slot.block, v, depth) : false;
  }
  function valOf(M, id, v, depth) {
    var b = M.B[id];
    if (!b || (depth || 0) > 40) return 0;
    var d = PS.DEFS[b.type];
    if (d.read) return d.read(v);
    var args = b.s.map(function (s) { return numOf(M, s, v, (depth || 0) + 1); });
    args.push(b.f);
    var x = d.fn ? d.fn.apply(null, args) : 0;
    return isFinite(x) ? x : 0;
  }
  function boolOf(M, id, v, depth) {
    var b = M.B[id];
    if (!b || (depth || 0) > 40) return false;
    var d = PS.DEFS[b.type];
    if (d.test) return d.test(v);
    if (d.fnb) return !!d.fnb.apply(null, b.s.map(function (s) { return boolSlot(M, s, v, (depth || 0) + 1); }));
    if (d.fn) return !!d.fn.apply(null, b.s.map(function (s) { return numOf(M, s, v, (depth || 0) + 1); }));
    return false;
  }
  function runList(M, list, v, tr, depth) {
    var sum = 0;
    if ((depth || 0) > 20) return 0;
    for (var i = 0; i < list.length; i++) {
      var b = M.B[list[i]];
      if (!b) continue;
      if (b.type === "plus" || b.type === "minus") {
        var x = numOf(M, b.s[0], v, 0);
        if (b.type === "minus") x = -x;
        sum += x;
        if (tr) tr[b.id] = x;
      } else if (b.type === "if" || b.type === "ifelse") {
        var c = boolSlot(M, b.s[0], v, 0);
        if (c) sum += runList(M, b.bodies[0], v, tr, (depth || 0) + 1);
        else if (b.type === "ifelse") sum += runList(M, b.bodies[1], v, tr, (depth || 0) + 1);
      }
    }
    return sum;
  }

  /* ── 글로 옮기기 (기록 표와 도움말에 씀) ── */
  function num(x) { return String(Math.round(x * 1000) / 1000); }
  function slotText(M, slot, spec) {
    if (slot.block) return blockText(M, slot.block);
    return spec.s === "bool" ? "?" : num(slot.v);
  }
  function blockText(M, id) {
    var b = M.B[id];
    if (!b) return "?";
    var d = PS.DEFS[b.type], out = [], k = 0;
    d.tpl.forEach(function (p) {
      if (typeof p === "string") out.push(p);
      else if (p.s) { out.push(slotText(M, b.s[k], p)); k++; }
      else if (p.dd) {
        var lab = "";
        p.opts.forEach(function (o) { if (String(o[0]) === String(b.f[p.dd])) lab = o[1]; });
        out.push(lab);
      }
    });
    var t = out.join(" ").replace(/ 값/g, "").replace(/\s+/g, " ").trim();
    return d.shape === "val" && b.s.length > 1 ? "(" + t + ")" : t;
  }
  function listText(M, list) {
    return list.map(function (id) {
      var b = M.B[id];
      if (!b) return "";
      if (b.type === "plus") return "+ " + slotText(M, b.s[0], { s: "num" });
      if (b.type === "minus") return "− " + slotText(M, b.s[0], { s: "num" });
      if (b.type === "if") return "만일 " + slotText(M, b.s[0], { s: "bool" }) + " 이면 { " + listText(M, b.bodies[0]) + " }";
      if (b.type === "ifelse") {
        return "만일 " + slotText(M, b.s[0], { s: "bool" }) + " 이면 { " + listText(M, b.bodies[0]) +
          " } 아니면 { " + listText(M, b.bodies[1]) + " }";
      }
      return blockText(M, id);
    }).filter(Boolean).join("  ");
  }
  function countScore(M, list) {
    var n = 0;
    list.forEach(function (id) {
      var b = M.B[id];
      if (!b) return;
      if (b.type === "plus" || b.type === "minus") n++;
      (b.bodies || []).forEach(function (body) { n += countScore(M, body); });
    });
    return n;
  }

  function blocks(M) {
    var list = M.scripts.reward;
    var text = listText(M, list), n = countScore(M, list);
    function fn(phi, ald, thd, u, th) {
      return runList(M, list, { phi: phi, ald: ald, thd: thd, u: u, th: th || 0 }, null, 0);
    }
    function trace(phi, ald, thd, u, th) {
      var tr = {};
      runList(M, list, { phi: phi, ald: ald, thd: thd, u: u, th: th || 0 }, tr, 0);
      return tr;
    }
    return {
      kind: "blocks", ok: n > 0,
      text: text ? "점수 = " + text : "아직 점수 블록이 없습니다",
      label: text || "빈 규칙", key: "blocks:" + text, fn: fn, trace: trace
    };
  }

  /* ── 시작하기 아래 블록 읽기 ── */
  function runOpts(M) {
    var o = { gens: 24, pop: 20, sigma: 0.06, learn: false, test: null };
    (M.scripts.run || []).forEach(function (id) {
      var b = M.B[id];
      if (!b) return;
      var x = b.s.length ? numOf(M, b.s[0], { phi: 0, ald: 0, thd: 0, u: 0, th: 0 }, 0) : 0;
      if (b.type === "cond") {
        if (b.f.key === "gens") o.gens = Math.max(4, Math.min(80, Math.round(x)));
        if (b.f.key === "pop") o.pop = Math.max(6, Math.min(40, Math.round(x)));
        if (b.f.key === "sigma") o.sigma = Math.max(1, Math.min(30, x)) / 100;
      }
      if (b.type === "learn") o.learn = true;
      if (b.type === "test" && o.test === null) o.test = Math.max(3, Math.min(20, Math.round(x)));
    });
    return o;
  }

  /* ── 규칙 점검: 학습하기 전에 흔한 실수를 잡습니다 ── */
  function check(rule) {
    if (!rule.ok) return { level: "none", msg: "점수 블록이 없습니다. 점수 범주에서 블록을 끌어다 놓으세요." };
    var up = rule.fn(0, 0, 0, 0, 0), hang = rule.fn(Math.PI, 0, 0, 0, 0), spin = 0, n = 12;
    for (var i = 0; i < n; i++) spin += rule.fn(PS.wrap(i / n * 2 * Math.PI), 20, 5, 1, 0) / n;
    if (up <= hang) {
      return { level: "warn", msg: "매달려 있을 때 점수(" + num(hang) + ")가 똑바로 섰을 때(" + num(up) +
        ")보다 높거나 같아요. 더하기와 빼기를 확인해 보세요." };
    }
    if (spin > up) {
      return { level: "warn", msg: "진자를 빨리 돌릴 때 점수가 가장 높아요. 키트가 안전 정지할 수 있어요." };
    }
    return { level: "ok", msg: "똑바로 섰을 때 점수(" + num(up) + ")가 매달렸을 때(" + num(hang) + ")보다 높습니다." };
  }

  PS.reward = {
    basic: basic, blocks: blocks, runOpts: runOpts, check: check, WORD: WORD,
    blockText: blockText, valOf: valOf, boolOf: boolOf
  };
})(window.PS = window.PS || {});
