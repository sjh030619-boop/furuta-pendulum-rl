/* 실시간 학습 (CEM, cross-entropy method) + 안전 시험.
 *
 * 왜 CEM인가: 학습지가 설명하는 방식 그대로이기 때문입니다 — "스스로 여러 번 해 보고, 받은
 * 점수로 알아낸다". 한 세대마다 정책을 조금씩 바꿔 여러 개 만들어 보고, 점수가 높았던 것들만
 * 남겨 그쪽으로 옮깁니다. 기울기도, 별도 런타임도 필요 없습니다.
 *
 * 학생이 블록으로 새 규칙을 만들면 미리 학습된 정책에서 출발해 수십 초 동안 다듬습니다.
 * 처음부터 배우는 것이 아니라 "고쳐 배우는" 것이고, 그래서 수업 중에 기다릴 만합니다.
 *
 * 실기에서도 버티게 하는 두 장치 (s2r/cem_finetune.py 에서 먼저 시험한 방식 그대로)
 *   조건 섞기  시작 자세 다섯 개를 서로 다른 조건 다섯 개(공칭, 마찰 큼, 마찰 작음, 토크 약함+지연, 결합 약함)와
 *             하나씩 짝지어 점수를 냅니다. 계산량은 같습니다. 공칭 하나에서만 점수를 내면 다른 조건에서 못 세우는
 *             정책이 나옵니다(128칸 중 120 이상 유지가 11개 규칙 중 2개 → 조건을 섞으면 7개).
 *   안전 시험  학생 점수와 상관없이 같은 40칸 시험(조건 8개 x 시작 5개, 12초 안에 세워 ±5° 안에서 5초 버팀)을 봅니다.
 *             시작 다섯 개에는 '완전히 매달려 멈춘 상태'가 들어 있습니다. 실제 키트가 바로 그 상태에서 시작하기 때문입니다
 *             (이 시작이 없던 때에는, 멈춘 상태에서 아예 움직이지 않는 정책이 시험을 통과했습니다).
 *             다 배운 뒤 후보 넷(점수 최고였던 것, 마지막 평균, 중간 평균 둘) 가운데 37칸 이상 통과한 것 중에서
 *             점수가 가장 높은 것을 고르고, 하나도 없거나 바탕보다 점수가 낮으면 바탕 정책을 그대로 씁니다.
 *
 * 안전 정지(팔·진자 속도 한계)로 끊기면, 남은 시간은 '진자가 매달려 멈춘 상태'의 점수로 채웁니다.
 * 실기에서 안전 정지 뒤 실제로 그렇게 되고, 이렇게 안 하면 감점만 있는 규칙에서 일찍 끊기는 쪽이 이득이 됩니다.
 */
(function (PS) {
  "use strict";

  var STARTS = [
    [0.0, 0.00], [0.0, 0.35], [0.0, -0.35],
    [0.2, 0.10], [-0.2, -0.10]
  ];
  var DEG5 = 5 * Math.PI / 180;

  function gauss(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* 한 정책을 고정된 시작 자세들에서 돌려 평균 점수를 냅니다. sims[i] 가 i 번째 시작 자세의 조건입니다. */
  function score(sims, mlp, rule, steps) {
    var total = 0, obs = new Float64Array(PS.OBS_N);
    for (var s = 0; s < STARTS.length; s++) {
      var sim = sims[s % sims.length];
      sim.reset(STARTS[s][0], STARTS[s][1], 0, 0);
      for (var i = 0; i < steps; i++) {
        sim.obs(obs);
        var u = mlp.forward(obs);
        sim.step(u);
        total += rule.fn(sim.phi(), sim.ald, sim.thd, u, sim.th);
        if (sim.dead) {
          total += (steps - i - 1) * rule.fn(Math.PI, 0, 0, 0, sim.th);
          break;
        }
      }
    }
    return total / STARTS.length;
  }

  /* 안전 시험 한 칸: secs 안에 |기울기| ≤ 5°, |진자 속도| ≤ 0.5 rad/s 를 hold 초 연속. 세우면 유지 시작 시각, 아니면 -1 */
  function strictEpisode(sim, mlp, th0, al0, secs, hold) {
    var obs = new Float64Array(PS.OBS_N), n = Math.round(secs / sim.ctrlDt), need = Math.round(hold / sim.ctrlDt), run = 0;
    sim.reset(th0, al0, 0, 0);
    for (var i = 0; i < n; i++) {
      sim.obs(obs);
      sim.step(mlp.forward(obs));
      var phi = sim.phi();
      run = (phi <= DEG5 && phi >= -DEG5 && sim.ald <= 0.5 && sim.ald >= -0.5) ? run + 1 : 0;
      if (run >= need) return (i + 1 - need) * sim.ctrlDt;
      if (sim.dead) return -1;
    }
    return -1;
  }

  /* 안전 시험: 조건 8개 x 시작 5개 = 40칸 (full 이면 파이썬 평가와 같은 시작 16개 = 128칸). 통과한 칸 수를 돌려줍니다. */
  function gate(C, mlp, full, cells) {
    var G = C.gate, starts = full ? G.all_starts : G.starts, pass = 0;
    for (var c = 0; c < G.conds.length; c++) {
      var sim = new PS.Furuta(C, G.conds[c].over);
      for (var s = 0; s < starts.length; s++) {
        var ok = strictEpisode(sim, mlp, starts[s][0], starts[s][1], G.secs, G.hold) >= 0;
        if (ok) pass++;
        if (cells) cells.push(ok ? 1 : 0);
      }
    }
    return pass;
  }

  function Trainer(C, base, rule, opt) {
    opt = opt || {};
    this.C = C;
    this.sims = (opt.robust === false ? [{}] : C.robust).map(function (over) { return new PS.Furuta(C, over); });
    this.rule = rule;
    this.steps = Math.round((opt.episodeSec || C.episode_sec) / this.sims[0].ctrlDt);
    this.gens = opt.gens || 24;
    this.pop = opt.pop || 20;
    this.elite = opt.elite || 5;
    this.sigma = opt.sigma || 0.06;
    this.decay = opt.decay || 0.93;
    this.rng = mulberry(opt.seed || 12345);
    this.base = base;
    this.best = base.clone();
    this.work = base.clone();
    this.mean = base.flatten();
    this.gen = 0;
    this.history = [];
    this.bestScore = -Infinity;
    this.done = false;
    this.mids = [];
    this.baseScore = score(this.sims, base, rule, this.steps);
  }

  Trainer.prototype.generation = function () {
    var n = this.mean.length, cand = [], i, k;
    for (i = 0; i < this.pop; i++) {
      var v = new Float64Array(n);
      for (k = 0; k < n; k++) v[k] = this.mean[k] + this.sigma * gauss(this.rng);
      this.work.load(v);
      cand.push({ v: v, s: score(this.sims, this.work, this.rule, this.steps) });
    }
    cand.sort(function (a, b) { return b.s - a.s; });
    var next = new Float64Array(n);
    for (i = 0; i < this.elite; i++) for (k = 0; k < n; k++) next[k] += cand[i].v[k] / this.elite;
    this.mean = next;
    this.sigma *= this.decay;
    if (cand[0].s > this.bestScore) { this.bestScore = cand[0].s; this.best.load(cand[0].v); }
    this.history.push(cand[0].s);
    this.gen++;
    if (this.gen === Math.floor(this.gens / 3) || this.gen === Math.floor(2 * this.gens / 3)) this.mids.push(Float64Array.from(next));
    if (this.gen >= this.gens) this.done = true;
    return cand[0].s;
  };

  /* 다 배운 뒤: 후보마다 점수와 안전 시험을 보고 하나를 고릅니다. 후보 하나씩 나눠 부를 수 있게 단계로 나눴습니다. */
  Trainer.prototype.candidates = function () {
    var self = this, list = [{ name: "점수 최고 후보", mlp: this.best }];
    var m = this.base.clone();
    m.load(this.mean);
    list.push({ name: "마지막 평균", mlp: m });
    this.mids.forEach(function (v, i) {
      var c = self.base.clone();
      c.load(v);
      list.push({ name: "중간 평균 " + (i + 1), mlp: c });
    });
    return list;
  };
  Trainer.prototype.judge = function (cand) {
    cand.score = score(this.sims, cand.mlp, this.rule, this.steps);
    cand.gate = gate(this.C, cand.mlp, false);
    return cand;
  };
  Trainer.prototype.choose = function (list) {
    var min = this.C.gate.pass_min, ok = list.filter(function (c) { return c.gate >= min; });
    var top = list.reduce(function (a, b) { return b.gate > a.gate ? b : a; }, list[0]);
    if (!ok.length) return { mlp: this.base.clone(), fallback: "gate", gate: top.gate, picked: "바탕 정책", score: this.baseScore };
    var best = ok.reduce(function (a, b) { return b.score > a.score ? b : a; }, ok[0]);
    if (best.score <= this.baseScore) {
      return { mlp: this.base.clone(), fallback: "noimprove", gate: best.gate, picked: "바탕 정책", score: this.baseScore };
    }
    return { mlp: best.mlp, fallback: null, gate: best.gate, picked: best.name, score: best.score };
  };

  /* 화면 주사율(requestAnimationFrame)에 묶지 않습니다. 창이 가려져 있으면 그쪽은 멈추거나 아주 느려지기 때문입니다.
   * MessageChannel 로 한 세대마다 양보하면 화면도 멈추지 않고, 가려진 창에서도 같은 속도로 배웁니다. */
  function yielder(fn) {
    if (typeof MessageChannel === "undefined") return function () { setTimeout(fn, 0); };
    var ch = new MessageChannel();
    ch.port1.onmessage = fn;
    return function () { ch.port2.postMessage(0); };
  }

  /* onProgress(p, history, trainer, phase): phase 는 "learn"(세대 진행) 또는 "gate"(안전 시험 중).
   * onDone(result, trainer): result = { mlp, fallback: null | "gate" | "noimprove", gate, picked, score } */
  function run(C, base, rule, opt, onProgress, onDone) {
    var tr = new Trainer(C, base, rule, opt), list = null, at = 0;
    var next = yielder(function () {
      if (tr.stopped) return;
      if (!tr.done) {
        tr.generation();
        if (onProgress) onProgress(tr.gen / tr.gens, tr.history, tr, "learn");
        next();
        return;
      }
      if (!list) list = tr.candidates();
      if (at < list.length) {
        tr.judge(list[at++]);
        if (onProgress) onProgress(1, tr.history, tr, "gate");
        next();
        return;
      }
      if (onDone) onDone(tr.choose(list), tr);
    });
    next();
    return tr;
  }

  PS.learn = { run: run, Trainer: Trainer, score: score, gate: gate, strictEpisode: strictEpisode, STARTS: STARTS };
})(window.PS = window.PS || {});
