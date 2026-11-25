/* 작은 선 그래프. 이번 시도와 지난번 시도를 겹쳐 보여 주는 데 씁니다.
 *
 *   var c = new PS.Chart(host, { xlabel: "초", ymin: 0, ymax: 100, digits: 0 });
 *   c.set([{ name: "지난번", color, dash, x: [...], y: [...] }, { name: "이번", color, x, y }], xmax);
 *
 * 크기는 그릴 때 host 의 실제 픽셀 크기를 재서 정합니다(글자가 늘어나지 않게). 탭이 가려져 있으면 그리지 않으므로
 * 탭을 열 때와 창 크기가 바뀔 때 draw() 를 다시 부릅니다. 마우스를 올리면 그 시각의 값을 보여 줍니다.
 * ymin, ymax 를 주지 않으면 자료에 맞춥니다.
 */
(function (PS) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var INK = "#51606E", GRID = "#E1E7E9";

  function node(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function Chart(host, opt) {
    var self = this;
    this.host = host;
    this.opt = opt || {};
    this.series = [];
    this.xmax = null;
    host.innerHTML = "";
    this.svg = node("svg", { "class": "chartsvg", width: "100%", height: "100%" });
    host.appendChild(this.svg);
    this.tip = document.createElement("div");
    this.tip.className = "ctip";
    this.tip.hidden = true;
    host.appendChild(this.tip);
    host.addEventListener("pointermove", function (e) { self.hover(e); });
    host.addEventListener("pointerleave", function () { self.tip.hidden = true; if (self.cross) self.cross.setAttribute("visibility", "hidden"); });
  }

  Chart.prototype.set = function (series, xmax) {
    this.series = series || [];
    this.xmax = xmax || null;
    this.draw();
  };

  Chart.prototype.fmt = function (v) {
    var d = this.opt.digits === undefined ? 0 : this.opt.digits;
    return v.toFixed(d);
  };

  Chart.prototype.draw = function () {
    var W = this.host.clientWidth, H = this.host.clientHeight, svg = this.svg, o = this.opt;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (W < 60 || H < 50) return;
    var L = 36, R = 10, T = 8, B = 20, pw = W - L - R, ph = H - T - B;
    var xmax = this.xmax || 0, lo = Infinity, hi = -Infinity, any = false;
    this.series.forEach(function (s) {
      for (var i = 0; i < s.y.length; i++) {
        any = true;
        if (s.x[i] > xmax) xmax = s.x[i];
        if (s.y[i] < lo) lo = s.y[i];
        if (s.y[i] > hi) hi = s.y[i];
      }
    });
    if (!any) { lo = 0; hi = 1; }
    var ymin = o.ymin !== undefined ? o.ymin : lo, ymax = o.ymax !== undefined ? o.ymax : hi;
    if (o.ymax === undefined) { var pad = (ymax - ymin) * 0.08 || 0.5; ymax += pad; if (o.ymin === undefined) ymin -= pad; }
    if (ymax <= ymin) ymax = ymin + 1;
    if (!xmax) xmax = 1;
    var X = function (x) { return L + x / xmax * pw; };
    var Y = function (y) { return T + (1 - (y - ymin) / (ymax - ymin)) * ph; };
    this.geo = { L: L, pw: pw, T: T, ph: ph, xmax: xmax, X: X, Y: Y };

    for (var k = 0; k <= 2; k++) {
      var yv = ymin + (ymax - ymin) * k / 2, yy = Y(yv);
      svg.appendChild(node("line", { x1: L, x2: L + pw, y1: yy, y2: yy, stroke: GRID, "stroke-width": 1 }));
      svg.appendChild(node("text", { x: L - 5, y: yy + 3.5, "text-anchor": "end", "font-size": 10.5, fill: INK }, this.fmt(yv)));
    }
    for (k = 0; k <= 2; k++) {
      var xv = xmax * k / 2;
      svg.appendChild(node("text", { x: X(xv), y: H - 5, "text-anchor": k === 0 ? "start" : k === 2 ? "end" : "middle", "font-size": 10.5, fill: INK },
        (xv % 1 ? xv.toFixed(1) : String(xv)) + (k === 2 && o.xlabel ? " " + o.xlabel : "")));
    }
    if (!any) {
      svg.appendChild(node("text", { x: L + pw / 2, y: T + ph / 2 + 4, "text-anchor": "middle", "font-size": 12, fill: INK }, "아직 자료가 없습니다"));
    }
    this.series.forEach(function (s) {
      if (!s.y.length) return;
      var pts = [];
      for (var i = 0; i < s.y.length; i++) pts.push(X(s.x[i]).toFixed(1) + "," + Y(Math.max(ymin, Math.min(ymax, s.y[i]))).toFixed(1));
      var a = { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" };
      if (s.dash) a["stroke-dasharray"] = s.dash;
      svg.appendChild(node("polyline", a));
    });
    this.cross = node("line", { x1: 0, x2: 0, y1: T, y2: T + ph, stroke: "#14212B", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" });
    svg.appendChild(this.cross);
  };

  Chart.prototype.hover = function (e) {
    var g = this.geo;
    if (!g || !this.series.length) return;
    var r = this.host.getBoundingClientRect(), px = e.clientX - r.left;
    if (px < g.L || px > g.L + g.pw) { this.tip.hidden = true; this.cross.setAttribute("visibility", "hidden"); return; }
    var xv = (px - g.L) / g.pw * g.xmax, lines = [], self = this;
    this.series.forEach(function (s) {
      if (!s.x.length) return;
      var best = 0, bd = Infinity;
      for (var i = 0; i < s.x.length; i++) { var d = Math.abs(s.x[i] - xv); if (d < bd) { bd = d; best = i; } }
      if (bd > g.xmax * 0.06) return;
      lines.push('<span><i style="background:' + s.color + '"></i>' + s.name + " <b>" + self.fmt(s.y[best]) + "</b></span>");
    });
    if (!lines.length) { this.tip.hidden = true; return; }
    this.cross.setAttribute("x1", px);
    this.cross.setAttribute("x2", px);
    this.cross.setAttribute("visibility", "visible");
    this.tip.innerHTML = "<em>" + (xv % 1 ? xv.toFixed(1) : String(xv)) + (this.opt.xlabel ? " " + this.opt.xlabel : "") + "</em>" + lines.join("");
    this.tip.hidden = false;
    var left = px + 12;
    if (left + 110 > r.width) left = px - 118;
    this.tip.style.left = left + "px";
    this.tip.style.top = "6px";
  };

  PS.Chart = Chart;
})(window.PS = window.PS || {});
