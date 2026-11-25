/* 정책: 상태를 보고 모터 힘을 고르는 작은 신경망.
 *
 * s2r/export_app.py 가 rev11 실기 보정 환경에서 학습한 SB3 정책의 가중치를 그대로 내보낸 것을 읽습니다.
 * 블록으로 만든 규칙은 가운데 칸(sac_s1e1, 1단계에서 쓴 보상)에서 출발해 고쳐 배웁니다.
 * 층이 둘(64×64)뿐이라 행렬곱 세 번이면 끝나고, 별도 런타임이 필요 없습니다.
 *   off-policy(SAC 등): ReLU 두 층 → tanh 로 눌러서 −1…1
 *   on-policy(PPO 등):  tanh 두 층 → 그대로 쓰고 −1…1 로 자름
 */
(function (PS) {
  "use strict";

  function MLP(spec) {
    this.layers = spec.layers.map(function (l) {
      var rows = l.w.length, cols = l.w[0].length;
      var w = new Float64Array(rows * cols);
      for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) w[i * cols + j] = l.w[i][j];
      return { w: w, b: Float64Array.from(l.b), rows: rows, cols: cols };
    });
    this.act = spec.activation === "tanh" ? Math.tanh : function (x) { return x > 0 ? x : 0; };
    this.squash = spec.squash !== false;
    this.name = spec.name || "";
    this.algo = spec.algo || "";
    this.size = this.layers.reduce(function (s, l) { return s + l.w.length + l.b.length; }, 0);
    this._buf = this.layers.map(function (l) { return new Float64Array(l.rows); });
  }

  MLP.prototype.forward = function (obs) {
    var x = obs;
    for (var li = 0; li < this.layers.length; li++) {
      var L = this.layers[li], out = this._buf[li], last = li === this.layers.length - 1;
      for (var i = 0; i < L.rows; i++) {
        var s = L.b[i], off = i * L.cols;
        for (var j = 0; j < L.cols; j++) s += L.w[off + j] * x[j];
        out[i] = last ? s : this.act(s);
      }
      x = out;
    }
    var u = x[0];
    if (this.squash) u = Math.tanh(u);
    return u < -1 ? -1 : u > 1 ? 1 : u;
  };

  /* CEM이 건드릴 수 있도록 가중치를 하나의 벡터로 펴고 되돌립니다. */
  MLP.prototype.flatten = function () {
    var v = new Float64Array(this.size), k = 0, i;
    this.layers.forEach(function (L) {
      for (i = 0; i < L.w.length; i++) v[k++] = L.w[i];
      for (i = 0; i < L.b.length; i++) v[k++] = L.b[i];
    });
    return v;
  };

  MLP.prototype.load = function (v) {
    var k = 0, i;
    this.layers.forEach(function (L) {
      for (i = 0; i < L.w.length; i++) L.w[i] = v[k++];
      for (i = 0; i < L.b.length; i++) L.b[i] = v[k++];
    });
    return this;
  };

  MLP.prototype.clone = function () {
    var c = Object.create(MLP.prototype);
    c.layers = this.layers.map(function (L) {
      return { w: Float64Array.from(L.w), b: Float64Array.from(L.b), rows: L.rows, cols: L.cols };
    });
    c.act = this.act; c.squash = this.squash; c.name = this.name; c.algo = this.algo;
    c.size = this.size;
    c._buf = c.layers.map(function (L) { return new Float64Array(L.rows); });
    return c;
  };

  /* 번들된 정책 중 규칙에 맞는 것을 고릅니다. 기본 규칙은 조합마다 하나씩 미리 학습해 두었고,
   * 없으면 가장 가까운 것으로 대신합니다(그 경우 화면에 "비슷한 규칙으로 시작"이라고 알립니다). */
  function pick(rule) {
    var bank = PS.POLICIES || {};
    if (rule.kind === "basic") {
      var exact = bank["sac_" + rule.key];
      if (exact) return { spec: exact, exact: true };
      var best = null, bd = 1e9;
      Object.keys(bank).forEach(function (k) {
        var m = /^sac_s(\d)e(\d)$/.exec(k);
        if (!m) return;
        var d = Math.abs(+m[1] - rule.smooth) + Math.abs(+m[2] - rule.effort);
        if (d < bd) { bd = d; best = bank[k]; }
      });
      if (best) return { spec: best, exact: false };
    }
    var keys = Object.keys(bank);
    if (bank.sac_s1e1) return { spec: bank.sac_s1e1, exact: false };
    var pref = keys.filter(function (k) { return k.indexOf("sac") === 0; })[0]
      || keys.filter(function (k) { return k.indexOf("tqc") === 0 || k.indexOf("td3") === 0; })[0]
      || keys[0];
    return pref ? { spec: bank[pref], exact: false } : null;
  }

  PS.MLP = MLP;
  PS.pickPolicy = pick;
})(window.PS = window.PS || {});
