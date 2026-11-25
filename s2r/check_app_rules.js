/* 앱에 들어간 '따라 하기' 아홉 칸 정책을 앱의 JS 물리로 돌려 봅니다: 세우는 시각, 서 있을 때 모터 힘, 팔이 돌아간 정도.
 *   node s2r/check_app_rules.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const SRC = path.join(__dirname, "..", "app", "src");
const ctx = { window: {}, Math, Float64Array, console, setTimeout, performance: require("perf_hooks").performance };
ctx.window.PS = {};
ctx.PS = ctx.window.PS;
vm.createContext(ctx);
for (const f of ["assets/model.js", "assets/policies.js", "js/physics.js", "js/policy.js", "js/learn.js"]) {
  vm.runInContext(fs.readFileSync(path.join(SRC, f), "utf8"), ctx);
}
const PS = ctx.window.PS, C = PS.MODEL, W = ["없음", "약하게", "강하게"];
const GT = C.gate.conds.length * C.gate.starts.length;
console.log("칸        규칙                      시험/" + GT + "  128칸  세움 s  힘   팔(바퀴)");
for (let s = 0; s < 3; s++) for (let e = 0; e < 3; e++) {
  const spec = PS.POLICIES["sac_s" + s + "e" + e], mlp = new PS.MLP(spec);
  const sim = new PS.Furuta(C), obs = new Float64Array(PS.OBS_N);
  let up = null, held = 0, fsum = 0, fn = 0;
  sim.reset(0, 0, 0, 0);
  for (let i = 0; i < 500 && !sim.dead; i++) {
    sim.obs(obs);
    sim.step(mlp.forward(obs));
    const deg = Math.abs(sim.phi()) * 180 / Math.PI;
    if (deg < 17 && Math.abs(sim.ald) < 3) { held += sim.ctrlDt; if (up === null && held > 0.4) up = sim.t - 0.4; } else held = 0;
    if (up !== null && sim.t > up + 1 && deg < 45) { fsum += Math.abs(sim.u); fn++; }
  }
  console.log(("s" + s + "e" + e).padEnd(8) + ("빠르기 " + W[s] + ", 힘 " + W[e]).padEnd(18) + String(PS.learn.gate(C, mlp, false)).padStart(10) +
    String(PS.learn.gate(C, mlp, true)).padStart(7) + (up === null ? "못 세움" : up.toFixed(1)).padStart(8) +
    (fn ? (fsum / fn * 100).toFixed(0) : "-").padStart(5) + (Math.abs(sim.th) / (2 * Math.PI)).toFixed(1).padStart(8));
}
