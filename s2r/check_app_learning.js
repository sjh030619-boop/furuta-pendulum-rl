/* 앱에 들어간 고쳐 배우기(app/src/js/learn.js)를 예시 규칙 전부에 돌려 봅니다.
 *
 *   node s2r/check_app_learning.js [세대=24] [한 세대에 해 볼 수=20]
 *
 * 앱과 같은 코드(물리, 블록 점수 계산, CEM, 조건 섞기, 안전 시험)를 화면 없이 실행합니다.
 * 규칙마다: 규칙 점검 결과, 학습에 걸린 시간, 고른 후보, 안전 시험 32칸, 고른 정책의 128칸, 행동 변화를 봅니다.
 * 행동은 공칭 조건에서 매달린 상태로 시작해 12초 동안 본 것입니다(세운 시각, 서 있을 때 모터 힘, 팔이 돌아간 정도).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = path.join(__dirname, "..", "app", "src");
const perf = require("perf_hooks").performance;
const ctx = { window: {}, Math, Float64Array, console, setTimeout, performance: perf };
ctx.window.PS = {};
ctx.PS = ctx.window.PS;
vm.createContext(ctx);
for (const f of ["assets/model.js", "assets/policies.js", "js/physics.js", "js/blockdefs.js", "js/reward.js", "js/policy.js", "js/learn.js", "js/examples.js"]) {
  vm.runInContext(fs.readFileSync(path.join(SRC, f), "utf8"), ctx);
}
const PS = ctx.window.PS, C = PS.MODEL, DEFS = PS.DEFS;
const gens = +process.argv[2] || 24, pop = +process.argv[3] || 20;
const GT = C.gate.conds.length * C.gate.starts.length;

/* 블록 편집기(blocks.js)의 build 와 같은 일: 짧은 표기 → 블록 모델 */
function modelOf(nodes) {
  const M = { B: {}, scripts: { reward: [], run: [] }, next: 1 };
  function mk(type) {
    const b = { id: M.next++, type, s: [], f: {}, bodies: [] };
    DEFS[type].tpl.forEach((p) => {
      if (typeof p === "string") return;
      if (p.s) b.s.push({ block: null, v: p.v === undefined ? 0 : p.v });
      else if (p.dd) b.f[p.dd] = p.opts[0][0];
    });
    for (let i = 0; i < (DEFS[type].bodies || 0); i++) b.bodies.push([]);
    M.B[b.id] = b;
    return b;
  }
  function build(n) {
    const b = mk(n.t);
    (n.a || []).forEach((arg, i) => {
      if (typeof arg === "number") b.s[i].v = arg;
      else if (arg) b.s[i].block = build(arg);
    });
    if (n.f) Object.assign(b.f, n.f);
    (n.body || []).forEach((list, i) => { b.bodies[i] = list.map(build); });
    return b.id;
  }
  M.scripts.reward = nodes.map(build);
  return M;
}

function behave(mlp) {
  const sim = new PS.Furuta(C), obs = new Float64Array(PS.OBS_N);
  let up = null, held = 0, fs2 = 0, fn = 0;
  sim.reset(0, 0, 0, 0);
  for (let i = 0; i < 600 && !sim.dead; i++) {
    sim.obs(obs);
    sim.step(mlp.forward(obs));
    const deg = Math.abs(sim.phi()) * 180 / Math.PI;
    if (deg < 17 && Math.abs(sim.ald) < 3) { held += sim.ctrlDt; if (up === null && held > 0.4) up = sim.t - 0.4; } else held = 0;
    if (up !== null && sim.t > up + 1 && deg < 45) { fs2 += Math.abs(sim.u); fn++; }
  }
  return { up, force: fn ? fs2 / fn * 100 : null, arm: Math.abs(sim.th) / (2 * Math.PI), dead: sim.dead };
}

const base = new PS.MLP(PS.pickPolicy({ kind: "blocks" }).spec);
const b0 = behave(base);
const fmt = (x, d) => (x === null || x === undefined ? "-" : x.toFixed(d));
console.log("바탕 정책: 안전 시험 " + PS.learn.gate(C, base, false) + "/" + GT + ", 128칸 " + PS.learn.gate(C, base, true) +
  ", 세움 " + fmt(b0.up, 1) + "초, 모터 힘 " + fmt(b0.force, 0) + ", 팔 " + fmt(b0.arm, 1) + "바퀴");
console.log("설정: 세대 " + gens + ", 한 세대에 " + pop + "개\n");
console.log("규칙".padEnd(22) + "종류  규칙 점검  학습 s  결과".padEnd(40) + "시험   128칸   세움 s  힘   팔(바퀴)");

const rows = [];
for (const ex of PS.EXAMPLES) {
  const rule = PS.reward.blocks(modelOf(ex.nodes));
  const chk = PS.reward.check(rule);
  const t0 = perf.now();
  const tr = new PS.learn.Trainer(C, base, rule, { gens, pop, elite: Math.max(3, Math.round(pop / 4)) });
  while (!tr.done) tr.generation();
  const res = tr.choose(tr.candidates().map((c) => tr.judge(c)));
  const sec = (perf.now() - t0) / 1000;
  const full = res.fallback ? null : PS.learn.gate(C, res.mlp, true);
  const b = behave(res.mlp);
  const kind = ex.bad ? "틀림" : ex.hard ? "어려움" : "보통";
  const what = res.fallback === "gate" ? "탈락 → 바탕 유지" : res.fallback === "noimprove" ? "개선 없음 → 바탕 유지" : res.picked;
  rows.push({ name: ex.name, kind, check: chk.level, sec, fallback: res.fallback, gate: res.gate, full, up: b.up, force: b.force, arm: b.arm });
  console.log(ex.name.padEnd(22 - Math.floor(ex.name.replace(/[ -~]/g, "").length * 0.6)) + kind.padEnd(5) + (chk.level === "ok" ? "통과" : "경고").padEnd(7) +
    sec.toFixed(1).padStart(6) + "  " + what.padEnd(18 - Math.floor(what.replace(/[ -~]/g, "").length * 0.6)) + String(res.gate).padStart(5) + "  " +
    (full === null ? "-" : String(full)).padStart(6) + "  " + fmt(b.up, 1).padStart(6) + "  " + fmt(b.force, 0).padStart(3) + "  " + fmt(b.arm, 1).padStart(6));
}
const good = rows.filter((r) => r.kind !== "틀림"), bad = rows.filter((r) => r.kind === "틀림");
const passed = good.filter((r) => !r.fallback);
console.log("\n정상 규칙 " + good.length + "개 중 안전 시험 통과 " + passed.length + "개, 그중 128칸 120 이상 " + passed.filter((r) => r.full >= 120).length +
  "개 (최저 " + Math.min.apply(null, passed.map((r) => r.full)) + ")");
console.log("틀린 규칙 " + bad.length + "개 중 규칙 점검이 경고한 것 " + bad.filter((r) => r.check !== "ok").length + "개, 안전 시험이 걸러낸 것 " + bad.filter((r) => r.fallback === "gate").length + "개");
console.log("학습 시간: 평균 " + (rows.reduce((a, r) => a + r.sec, 0) / rows.length).toFixed(1) + "초, 최대 " + Math.max.apply(null, rows.map((r) => r.sec)).toFixed(1) + "초");
fs.writeFileSync(path.join(__dirname, "app_learning.json"), JSON.stringify({ gens, pop, base: b0, rows }, null, 1));
