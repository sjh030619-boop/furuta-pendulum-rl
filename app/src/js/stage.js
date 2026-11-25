/* 키트 화면: 비스듬히 위에서 본 모습. 초안의 렌더러를 그대로 가져오되, 시뮬레이션 내부
 * 객체가 아니라 장치가 보내 주는 상태({t, th, al, thd, ald, u})를 받도록 바꿨습니다.
 * 그래서 시뮬레이션이든 실제 키트든 같은 화면을 씁니다.
 *
 * 각도 규약: al = 0 아래로 매달림, al = π 도립. 렌더러 내부에서는 도립 기준(a = al − π)을 씁니다.
 * 치수는 실제 키트에 맞춘 mm 입니다.
 */
(function (PS) {
  "use strict";

  function Stage(root, opt) {
    opt = opt || {};
    root.appendChild(document.getElementById("stageTpl").content.cloneNode(true));
    var $ = function (q) { return root.querySelector(q); };
    var A = 92, HB = 46, T1 = 150, T2 = 248, H = 268, R = 88, L = 135;
    var S = 1.12, CX = 400, CY = 478, CE = 0.866, SE = 0.5;
    var phi = 0.62, running = false, showNums = !!opt.nums;
    var chips = root.querySelectorAll(".phase .ph");
    var last = { t: 0, th: 0, al: 0, thd: 0, ald: 0, u: 0 };
    var extra = {};
    var note = "";

    function P(X, Y, Z) {
      var c = Math.cos(phi), n = Math.sin(phi);
      var xr = X * c - Y * n, yr = X * n + Y * c;
      return [CX + S * xr, CY - S * Z * CE + S * yr * SE, yr];
    }
    function pts(list) {
      return list.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ");
    }
    function set(el, a) { for (var k in a) el.setAttribute(k, a[k]); }
    function cyl(r, z0, z1, side, top) {
      var b = P(0, 0, z0), t = P(0, 0, z1), rx = S * r, ry = S * r * SE;
      return '<path fill="' + side + '" d="M' + (CX - rx) + " " + t[1] + "L" + (CX - rx) + " " + b[1] +
        "A" + rx + " " + ry + " 0 0 0 " + (CX + rx) + " " + b[1] + "L" + (CX + rx) + " " + t[1] + 'Z"/>' +
        '<ellipse cx="' + CX + '" cy="' + t[1] + '" rx="' + rx + '" ry="' + ry + '" fill="' + top + '"/>';
    }
    function drawStatic() {
      var F = 240;
      $(".floor").setAttribute("points", pts([P(-F, -F, 0), P(F, -F, 0), P(F, F, 0), P(-F, F, 0)]));
      var cor = [[A, -A], [A, A], [-A, A], [-A, -A]], nrm = [[1, 0], [0, 1], [-1, 0], [0, -1]], html = "";
      for (var i = 0; i < 4; i++) {
        var ny = nrm[i][0] * Math.sin(phi) + nrm[i][1] * Math.cos(phi);
        if (ny <= 0) continue;
        var c1 = cor[i], c2 = cor[(i + 1) % 4];
        var shade = Math.abs(nrm[i][0] * Math.cos(phi) - nrm[i][1] * Math.sin(phi)) > 0.6 ? "#15181B" : "#1E2226";
        html += '<polygon fill="' + shade + '" points="' +
          pts([P(c1[0], c1[1], 0), P(c2[0], c2[1], 0), P(c2[0], c2[1], HB), P(c1[0], c1[1], HB)]) + '"/>';
      }
      html += '<polygon fill="#2A2F35" points="' + pts(cor.map(function (c) { return P(c[0], c[1], HB); })) + '"/>';
      $(".base").innerHTML = html;
      $(".tower").innerHTML = cyl(62, HB, HB + 14, "#16181B", "#23272C") + cyl(54, HB + 14, T1, "#16181B", "#262A2F") +
        cyl(42, T1, T2, "#16181B", "#262A2F") + cyl(27, T2, T2 + 8, "#2F6B3C", "#4F9A5B");
      var ring = [];
      for (var k = 0; k <= 48; k++) {
        var t = k / 48 * 2 * Math.PI;
        ring.push(P(R * Math.cos(t), R * Math.sin(t), H));
      }
      $(".orbit").setAttribute("points", pts(ring));
    }
    function num(n, at, dx, dy) {
      return '<line x1="' + at[0] + '" y1="' + at[1] + '" x2="' + (at[0] + dx) + '" y2="' + (at[1] + dy) +
        '" stroke="#14212B" stroke-width="1.5"/><circle cx="' + (at[0] + dx) + '" cy="' + (at[1] + dy) +
        '" r="15" fill="#FFDC91" stroke="#14212B" stroke-width="1.5"/><text x="' + (at[0] + dx) + '" y="' +
        (at[1] + dy + 6) + '" text-anchor="middle" font-size="17" font-weight="800" fill="#14212B">' + n + "</text>";
    }

    function draw() {
      var s = last, a = s.al - Math.PI;
      var ct = Math.cos(s.th), sn = Math.sin(s.th), px = R * ct, py = R * sn;
      var pv = P(px, py, H), sa = Math.sin(a), ca = Math.cos(a);
      var tx = px - L * sa * sn, ty = py + L * sa * ct, tz = H + L * ca;
      var tip = P(tx, ty, tz), top = P(px, py, H + L), sh = P(tx, ty, 0);
      var h0 = P(-12 * ct, -12 * sn, H), h1 = P(26 * ct, 26 * sn, H), sen = P(-34 * ct, -34 * sn, H);
      ["armO", "arm"].forEach(function (c) { set($("." + c), { x1: h1[0], y1: h1[1], x2: pv[0], y2: pv[1] }); });
      ["rodO", "rod"].forEach(function (c) { set($("." + c), { x1: pv[0], y1: pv[1], x2: tip[0], y2: tip[1] }); });
      set($(".holder"), { x1: h0[0], y1: h0[1], x2: h1[0], y2: h1[1] });
      set($(".sensor"), { cx: sen[0], cy: sen[1] });
      set($(".ghost"), { x1: pv[0], y1: pv[1], x2: top[0], y2: top[1] });
      set($(".bob"), { cx: tip[0], cy: tip[1] });
      set($(".pivot"), { cx: pv[0], cy: pv[1] });
      set($(".shadow"), { cx: sh[0], cy: sh[1] });

      var rodg = $(".rodg"), tower = $(".tower"), armg = $(".armg"), svg = rodg.parentNode;
      if (tip[2] < 0 && tz < T2) { if (rodg.nextElementSibling !== tower) svg.insertBefore(rodg, tower); }
      else if (rodg.nextElementSibling !== armg) svg.insertBefore(rodg, armg);

      $(".nums").innerHTML = (showNums && !running)
        ? num(1, P(0, 0, 110), -120, 20) + num(2, P(px * 0.62, py * 0.62, H), 60, -70) +
          num(3, [(pv[0] + tip[0]) / 2, (pv[1] + tip[1]) / 2], 80, 10) + num(4, sen, -70, -50)
        : "";

      var deg = Math.abs(PS.wrap(a) * 180 / Math.PI);
      $(".deg").textContent = deg.toFixed(0) + "°";
      $(".spd").textContent = (Math.abs(s.ald) / 6.283).toFixed(1) + "바퀴/초";
      var ub = $(".ubar"), f = Math.max(-1, Math.min(1, s.u));
      ub.style.width = Math.abs(f) * 50 + "%";
      ub.style.left = (f < 0 ? 50 + f * 50 : 50) + "%";
      var rw = extra.reward === undefined ? 0 : extra.reward;
      $(".rw").textContent = Math.abs(rw) >= 10 ? rw.toFixed(0) : rw.toFixed(2);
      var v = extra.value === undefined ? 0 : extra.value;
      $(".vbar").style.width = Math.max(0, Math.min(1, v)) * 100 + "%";
      var tend = extra.tend || 10;
      $(".time").textContent = s.t.toFixed(1);
      $(".tend").textContent = tend.toFixed(1);
      $(".tbar").style.width = Math.min(100, s.t / tend * 100) + "%";
      $(".gfill").style.height = ((180 - deg) / 180 * 100) + "%";
      var peak = extra.peak === undefined ? 0 : extra.peak;
      $(".gpeak").style.bottom = (peak / 180 * 100) + "%";
      $(".gpeak").hidden = !(s.t > 0);
      $(".gpeak span").style.visibility = peak > 165 ? "hidden" : "visible";
      $(".res").textContent = extra.result || (s.t > 0 ? "시험 중…" : note);

      var ph = -1;
      if (s.t > 0) {
        if (deg < 26 && Math.abs(s.ald) < 6) ph = 2;
        else if (peak > 90) ph = 1;
        else ph = 0;
      }
      chips.forEach(function (c, i) { c.className = "ph" + (i === ph ? " on" : ""); });
    }

    this.label = function (text) { $(".where").textContent = text; };
    this.busy = function (on) { running = on; };
    this.show = function (state, ex) {
      if (state) last = state;
      if (ex) extra = ex;
      draw();
    };
    /* 시험이 시작되기 전에 보여 줄 한 줄 (키트 준비 단계). 시험이 시작되면 "시험 중…"으로 바뀝니다. */
    this.note = function (text) { note = text || ""; if (!extra.result && !(last.t > 0)) $(".res").textContent = note; };
    this.result = function (text) { extra.result = text || ""; $(".res").textContent = extra.result; };
    this.clearResult = function () { extra.result = ""; note = ""; $(".res").textContent = ""; };

    var view = $(".view"), dragX = null;
    view.addEventListener("pointerdown", function (e) { dragX = e.clientX; view.setPointerCapture(e.pointerId); });
    view.addEventListener("pointermove", function (e) {
      if (dragX === null) return;
      phi += (e.clientX - dragX) * 0.008;
      dragX = e.clientX;
      drawStatic();
      draw();
    });
    view.addEventListener("pointerup", function () { dragX = null; });

    drawStatic();
    draw();
  }

  PS.Stage = Stage;
})(window.PS = window.PS || {});
