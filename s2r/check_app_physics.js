/* 앱의 JS 물리(app/src/js/physics.js)가 파이썬 실기 보정 환경(s2r_env.py)과 같은 결과를 내는지 확인합니다.
 *
 *   python s2r/export_app.py --model      (app_ref.json 을 만듦)
 *   node s2r/check_app_physics.js
 *
 * 1. 열린 고리: 같은 행동 열을 넣었을 때 진자·팔 각도가 얼마나 벌어지나 (0.5 s, 1 s, 2 s, 3 s 시점까지의 최대 차이)
 * 2. 닫힌 고리: 바탕 정책 셋의 128칸 시험 결과가 파이썬과 몇 칸 같은가
 * 스윙업은 초기 조건에 민감해서 칸 하나하나가 모두 같을 수는 없습니다. 통과 수와 대부분의 칸이 같으면 됩니다.
 * 3. 속도: 제어 스텝 하나에 걸리는 시간 (앱 학습 시간 계산용)
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const SRC = path.join(ROOT, "app", "src");
const ctx = { window: {}, Math, Float64Array, console, MessageChannel: undefined, setTimeout, performance: require("perf_hooks").performance };
ctx.window.PS = {};
ctx.PS = ctx.window.PS;
vm.createContext(ctx);
for (const f of ["assets/model.js", "js/physics.js", "js/policy.js", "js/learn.js"]) {
  vm.runInContext(fs.readFileSync(path.join(SRC, f), "utf8"), ctx);
}
const PS = ctx.window.PS;
const C = PS.MODEL;
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "app_ref.json"), "utf8"));
const deg = (x) => x * 180 / Math.PI;
let bad = 0;

console.log("1. 열린 고리 (같은 행동 열, 최대 각도 차이 [도])");
console.log("   이름             0.5 s까지   1 s까지   2 s까지   3 s까지   (진자 / 팔)");
for (const q of ref.sequences) {
  const sim = new PS.Furuta(C, q.over);
  sim.reset(q.x0[0], q.x0[1], 0, 0);
  const marks = [25, 50, 100, 150], out = [];
  let dAl = 0, dTh = 0, dObs = 0;
  for (let i = 0; i < q.u.length; i++) {
    sim.step(q.u[i]);
    const s = q.states[i];
    dAl = Math.max(dAl, Math.abs(sim.al - s[1]));
    dTh = Math.max(dTh, Math.abs(sim.th - s[0]));
    if (i < 25) { const o = sim.obs(); for (let k = 0; k < 6; k++) dObs = Math.max(dObs, Math.abs(o[k] - q.obs[i + 1][k])); }
    if (marks.includes(i + 1)) out.push(deg(dAl).toFixed(2) + " / " + deg(dTh).toFixed(2));
  }
  console.log("   " + q.name.padEnd(14) + out.map((x) => x.padStart(14)).join("") + "   관측 차이(0.5 s) " + dObs.toExponential(1));
  const first = parseFloat(out[0]);
  if (first > 1.0) bad++;
}

console.log("\n2. 닫힌 고리 (바탕 정책, 128칸 시험)");
for (const name of Object.keys(ref.gates)) {
  const g = ref.gates[name];
  const mlp = new PS.MLP({ layers: g.layers, activation: "relu", squash: true });
  const cells = [];
  const total = PS.learn.gate(C, mlp, true, cells);
  let same = 0;
  for (let i = 0; i < cells.length; i++) if (cells[i] === g.cells[i]) same++;
  console.log("   " + name.padEnd(14) + " 파이썬 " + g.total + "/128   JS " + total + "/128   같은 칸 " + same + "/128");
  if (Math.abs(total - g.total) > 6) bad++;
}

console.log("\n3. 속도");
{
  const g = ref.gates[Object.keys(ref.gates)[0]];
  const mlp = new PS.MLP({ layers: g.layers, activation: "relu", squash: true });
  const sim = new PS.Furuta(C), obs = new Float64Array(PS.OBS_N);
  const n = 200000, t0 = ctx.performance.now();
  sim.reset(0, 0, 0, 0);
  for (let i = 0; i < n; i++) { sim.obs(obs); sim.step(mlp.forward(obs)); if (sim.dead || i % 600 === 599) sim.reset(0, 0, 0, 0); }
  const us = (ctx.performance.now() - t0) * 1000 / n;
  const cem = 24 * 20 * 5 * Math.round(C.episode_sec * C.ctrl_hz);
  console.log("   제어 스텝 하나 " + us.toFixed(2) + " µs → 기본 학습(24세대 x 20개 x 5번 x " + C.episode_sec + "초 = " + cem + " 스텝) 약 " + (cem * us / 1e6).toFixed(1) + "초, 안전 시험 32칸 약 " + (32 * 600 * us / 1e6).toFixed(2) + "초");
}
console.log(bad ? "\n결과: 확인 필요 (" + bad + "건)" : "\n결과: 통과");
process.exit(bad ? 1 : 0);
