/* 회전 도립 진자 rev11 의 동역학과 실기 경로.
 *
 * 파이썬의 실기 보정 환경(s2r/s2r_env.py, S2REnv)과 같은 계산을 합니다. 상수는 s2r/export_app.py 가 MuJoCo 모델과
 * 조원 측정값에서 뽑아 assets/model.js 로 내보내고, s2r/check_app_physics.js 가 두 쪽의 결과를 비교합니다.
 *
 * 동역학: 질량행렬은 진자 각도 a 에만 의존합니다.
 *   M00(a) = c0 + c1 cos a + c2 sin a + c3 cos 2a + c4 sin 2a + J      (J = 팔 축 모터 관성, 조건마다 다름)
 *   M01(a) = d0 + d1 cos a + d2 sin a                                  (결합 배율 k 에 선형)
 *   M11    = 상수,   중력 = g0 + g1 cos a + g2 sin a
 *   코리올리: h0 = M00' th' a' + M01' a'^2,  h1 = -0.5 M00' th'^2
 * 마찰: 팔 = 쿨롱 Fc±·tanh(w/0.05) + 점성 b·w,  진자 = 감쇠 + 쿨롱 + 공기저항 c2·w|w|
 * 적분: MuJoCo 와 같은 반암시 오일러(속도 먼저, 그 속도로 각도), 2 ms.
 *
 * 토크 경로 (실기 호스트 run_policy.py 와 같은 순서):
 *   u·2.52 → 속도-토크 제한(16 rad/s) → 상한 → 제동 전력 5 W(직립 ±20° 밖) → 0.01 A 양자화 → 실제 토크 = 명령 x kt
 *   명령은 관측 뒤 lat 서브스텝(0~4 ms) 지나서 적용됩니다.
 * 관측 6개: [cos th, sin th, cos al, sin al, thd/20, ald/40]
 *   th, thd 는 직전 주기의 모터 응답(약 한 주기 전 값), al 은 지금 센서 값(12비트), ald 는 20 ms 차분.
 * 안전 정지: 팔 > 1100 dps 또는 진자 > 25 rad/s 이면 dead = true (실기 호스트가 멈추는 조건).
 *
 * 조건(params): J, Fp, Fm, b, k, pend_fl, zero_deg, lat, kt. 주지 않으면 MODEL.nominal.
 */
(function (PS) {
  "use strict";

  var TWO_PI = Math.PI * 2;
  PS.OBS_N = 6;

  function wrap(x) {
    var y = (x + Math.PI) % TWO_PI;
    if (y < 0) y += TWO_PI;
    return y - Math.PI;
  }
  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    var e = Math.exp(2 * x);
    return (e - 1) / (e + 1);
  }

  function Furuta(C, params) {
    this.C = C;
    this.sub = Math.max(1, Math.round(1 / (C.ctrl_hz * C.timestep)));
    this.dt = C.timestep;
    this.ctrlDt = this.sub * this.dt;
    this.m00 = new Float64Array(5);
    this.m01 = new Float64Array(3);
    this._obs = new Float64Array(PS.OBS_N);
    this.setParams(params);
    this.reset(0, 0, 0, 0);
  }

  /* 조건 바꾸기. over 에 준 값만 공칭에서 바꿉니다. */
  Furuta.prototype.setParams = function (over) {
    var C = this.C, n = C.nominal, p = {}, k, i;
    for (k in n) p[k] = n[k];
    if (over) for (k in over) p[k] = over[k];
    this.p = p;
    for (i = 0; i < 5; i++) this.m00[i] = C.m00[i] + p.k * C.m00_k[i];
    for (i = 0; i < 3; i++) this.m01[i] = C.m01[i] + p.k * C.m01_k[i];
    this.m00[0] += p.J;
    this.fl = C.pend_fl * p.pend_fl;
    this.zero = p.zero_deg * Math.PI / 180;
    this.lat = p.lat | 0;
    return this;
  };

  Furuta.prototype.reset = function (th, al, thd, ald) {
    this.th = th || 0;
    this.al = al || 0;
    this.thd = thd || 0;
    this.ald = ald || 0;
    this.t = 0;
    this.tau = 0;
    this.u = 0;
    this.dead = false;
    this.rTh = this.th;
    this.rThd = this.thd;
    this.alPrev = null;
    this.measure();
    this.alPrev = this.mAl;
    return this;
  };

  /* 호스트가 이번 주기에 보는 값 */
  Furuta.prototype.measure = function () {
    var C = this.C, q = C.as5600;
    var al = Math.round(wrap(this.al + this.zero) / q) * q;
    var ald = this.alPrev === null ? 0 : wrap(al - this.alPrev) / this.ctrlDt;
    this.alPrev = al;
    this.mTh = Math.round(this.rTh / C.th_q) * C.th_q;
    this.mThd = Math.round(this.rThd / C.thd_q) * C.thd_q;
    this.mAl = al;
    this.mAld = ald;
    var o = this._obs;
    o[0] = Math.cos(this.mTh); o[1] = Math.sin(this.mTh);
    o[2] = Math.cos(al); o[3] = Math.sin(al);
    o[4] = this.mThd / C.obs_scale.arm;
    o[5] = ald / C.obs_scale.pend;
  };

  /* 정책 출력 u → 모터가 받는 토크 명령 (run_policy.py 와 같은 순서) */
  Furuta.prototype.command = function (u) {
    var C = this.C, T = C.tau_max, thd = this.mThd, tau = u * T, lim;
    if (tau * thd > 0) {
      lim = T * Math.max(0, 1 - Math.abs(thd) / C.arm_no_load_speed);
      if (tau > lim) tau = lim; else if (tau < -lim) tau = -lim;
    }
    if (tau > T) tau = T; else if (tau < -T) tau = -T;
    var phi = wrap(this.mAl - Math.PI);
    if (Math.abs(phi) > C.brake_zone && tau * thd < 0 && thd !== 0) {
      lim = C.brake_w / Math.abs(thd);
      if (tau > lim) tau = lim; else if (tau < -lim) tau = -lim;
    }
    var iq = Math.round(tau / C.kt_amp / 0.01);
    if (iq > C.iq_max) iq = C.iq_max; else if (iq < -C.iq_max) iq = -C.iq_max;
    return iq * 0.01 * C.kt_amp;
  };

  /* 물리 한 스텝(2 ms). 팔 마찰과 진자 공기저항은 스텝 시작 속도로 정해 한 스텝 동안 유지합니다(파이썬 환경과 같음). */
  Furuta.prototype.substep = function (tau) {
    var C = this.C, p = this.p, c0 = this.m00, c1 = this.m01, g = C.g1;
    var al = this.al, thd = this.thd, ald = this.ald, dt = this.dt;
    var f0 = tau - ((thd > 0 ? p.Fp : p.Fm) * tanh(thd / C.arm_fric_eps) + p.b * thd);
    var f1 = -C.pend_c2 * ald * Math.abs(ald) - C.pend_damping * ald - this.fl * tanh(ald / C.pend_fl_eps);
    var ca = Math.cos(al), sa = Math.sin(al), c2 = 2 * ca * ca - 1, s2 = 2 * sa * ca;
    var m00 = c0[0] + c0[1] * ca + c0[2] * sa + c0[3] * c2 + c0[4] * s2;
    var m01 = c1[0] + c1[1] * ca + c1[2] * sa;
    var m11 = C.m11;
    var d00 = -c0[1] * sa + c0[2] * ca - 2 * c0[3] * s2 + 2 * c0[4] * c2;
    var d01 = -c1[1] * sa + c1[2] * ca;
    var b0 = f0 - (d00 * thd * ald + d01 * ald * ald);
    var b1 = f1 - (-0.5 * d00 * thd * thd + (g[0] + g[1] * ca + g[2] * sa));
    var det = m00 * m11 - m01 * m01;
    this.thd = thd + dt * (b0 * m11 - b1 * m01) / det;
    this.ald = ald + dt * (b1 * m00 - b0 * m01) / det;
    this.th += dt * this.thd;
    this.al += dt * this.ald;
  };

  /* 제어 한 스텝(20 ms). 안전 정지 조건에 걸리면 dead 가 됩니다. */
  Furuta.prototype.step = function (u) {
    u = u < -1 ? -1 : u > 1 ? 1 : u;
    this.u = u;
    var tauNew = this.command(u) * this.p.kt, tauOld = this.tau, lat = this.lat;
    for (var i = 0; i < this.sub; i++) {
      this.substep(i < lat ? tauOld : tauNew);
      if (i === 0) { this.rTh = this.th; this.rThd = this.thd; }
    }
    this.tau = tauNew;
    this.t += this.ctrlDt;
    if (Math.abs(this.thd) > this.C.arm_speed_lim || Math.abs(this.ald) > this.C.pend_speed_lim) this.dead = true;
    this.measure();
    return this;
  };

  /* 학습된 정책이 보는 값. 환경(s2r_env.py)·실기 호스트와 순서·눈금이 같습니다. */
  Furuta.prototype.obs = function (out) {
    out = out || new Float64Array(PS.OBS_N);
    for (var i = 0; i < PS.OBS_N; i++) out[i] = this._obs[i];
    return out;
  };

  Furuta.prototype.phi = function () {
    return wrap(this.al - Math.PI);
  };

  Furuta.prototype.upright = function () {
    var p = this.phi();
    return p > -0.1745 && p < 0.1745 && this.ald > -2 && this.ald < 2;
  };

  PS.wrap = wrap;
  PS.Furuta = Furuta;
})(window.PS = window.PS || {});
