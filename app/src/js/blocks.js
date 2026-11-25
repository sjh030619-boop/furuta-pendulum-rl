/* 블록 편집기 — 엔트리와 같은 모양과 조립 방식.
 *
 * 블록은 DOM 안에 그대로 중첩해 놓아 자리가 저절로 잡히게 하고, 모양(V 홈, 반원 끝, 모자의 원, 육각)은
 * 자리가 잡힌 뒤 크기를 재서 블록마다 SVG 경로로 그립니다(decorate). 치수는 엔트리 편집기의 블록 경로와 같습니다.
 *   쌓기 블록  M0 0 l6 6 l6 -6 H(W-r) a r r 0 0 1 0 H  H12 l-6 6 l-6 -6 z        (높이 28, r = 높이/2)
 *   모자 블록  왼쪽에 반지름 17 원, 몸통 높이 28, 아래 블록은 12 들여서 붙음
 *   감싸는 블록 위 막대 28 + 안쪽(들여쓰기 14) + 아래 막대 15
 *   판단 블록  육각, 끝 폭 = 높이 x 0.4
 *
 * 모델
 *   B[id]          = { id, type, s: [{ block, v }], f: { 고르는 칸 }, bodies: [[id...]] }
 *   scripts.reward = [쌓기 블록 id, ...]   매 순간 점수 매기기
 *   scripts.run    = [쌓기 블록 id, ...]   시작하기 버튼을 클릭했을 때
 *   pos            = { reward, run }       모자 블록 묶음의 자리 ({x, y}). null 이면 위에서부터 차례로 놓음
 *   loose          = [{ lid, x, y, ids }]  조립소에 따로 놓아 둔 블록 묶음. 모자 블록에 붙지 않았으므로 실행되지 않음
 * 따라 하기에서는 고정된 블록 묶음(guide)을 보여 주고, 고르는 칸만 바꿀 수 있습니다.
 */
(function (PS) {
  "use strict";

  var DEFS = PS.DEFS, HATS = PS.HATS, PALETTE = PS.PALETTE, CATS = PS.CATS;
  var NS = "http://www.w3.org/2000/svg";

  function Blocks(container, paletteEl, railEl, opt) {
    opt = opt || {};
    var self = this;
    var M = null, G = null, mode = "free";
    var cat = "score", locked = false, drag = null;

    var board = document.createElement("div");
    board.className = "board";
    var canvas = document.createElement("div");
    canvas.className = "canvas";
    board.appendChild(canvas);
    var ghost = document.createElement("div");
    ghost.className = "ghost";
    ghost.hidden = true;
    board.appendChild(ghost);
    var checkEl = document.createElement("div");
    checkEl.className = "rulecheck";
    container.appendChild(checkEl);
    var trash = document.createElement("div");
    trash.className = "trash";
    trash.innerHTML = '<svg width="40" height="48" viewBox="0 0 46 56" aria-hidden="true"><rect x="6" y="14" width="34" height="38" rx="5" fill="#dfe8f0" stroke="#aebfce" stroke-width="2"/><rect x="2" y="7" width="42" height="7" rx="3.5" fill="#c7d5e2" stroke="#aebfce" stroke-width="2"/><rect x="17" y="2" width="12" height="6" rx="2" fill="#c7d5e2" stroke="#aebfce" stroke-width="2"/><path d="M15 22v22M23 22v22M31 22v22" stroke="#aebfce" stroke-width="2.4" stroke-linecap="round"/></svg>';
    trash.title = "여기로 끌면 버립니다";
    container.appendChild(trash);
    container.appendChild(board);
    self.board = board;

    var zoom = 1, hist = [], hpos = -1;
    var zoomer = document.createElement("div");
    zoomer.className = "zoomer";
    zoomer.innerHTML = '<button data-z="out" title="작게" aria-label="작게">−</button><button data-z="reset" title="원래 크기"><b>100%</b></button><button data-z="in" title="크게" aria-label="크게">＋</button>';
    container.appendChild(zoomer);

    function cur() { return mode === "guide" ? G : M; }
    function changed() {
      if (mode === "free") commit();
      if (opt.onChange) opt.onChange();
    }

    /* ── 모델 ── */
    function fresh() { return { B: {}, scripts: { reward: [], run: [] }, pos: { reward: null, run: null }, loose: [], next: 1 }; }
    function mkIn(model, type) {
      var d = DEFS[type];
      var b = { id: model.next++, type: type, s: [], f: {}, bodies: [] };
      d.tpl.forEach(function (p) {
        if (typeof p === "string") return;
        if (p.s) b.s.push({ block: null, v: p.v === undefined ? 0 : p.v });
        else if (p.dd) b.f[p.dd] = p.opts[0][0];
      });
      for (var i = 0; i < (d.bodies || 0); i++) b.bodies.push([]);
      model.B[b.id] = b;
      return b;
    }
    function mk(type) { return mkIn(M, type); }

    /* 예시 규칙·기본 블록을 적는 짧은 표기: 숫자 또는 { t, a: [칸...], f: {}, body: [[...]] } */
    function build(model, node) {
      var b = mkIn(model, node.t);
      (node.a || []).forEach(function (arg, i) {
        if (!b.s[i]) return;
        if (typeof arg === "number") b.s[i].v = arg;
        else if (arg) b.s[i].block = build(model, arg);
      });
      if (node.f) for (var k in node.f) b.f[k] = node.f[k];
      (node.body || []).forEach(function (list, i) {
        if (b.bodies[i]) b.bodies[i] = list.map(function (n) { return build(model, n); });
      });
      return b.id;
    }
    var DEFAULT_REWARD = [
      { t: "plus", a: [{ t: "height" }] },
      { t: "if", a: [{ t: "upright" }], body: [[{ t: "plus", a: [50] }]] },
      { t: "minus", a: [{ t: "mul", a: [{ t: "force" }, 0.5] }] }
    ];
    var DEFAULT_RUN = [
      { t: "cond", a: [24], f: { key: "gens" } },
      { t: "learn" },
      { t: "test", a: [10] }
    ];
    function defaults() {
      M = fresh();
      M.scripts.reward = DEFAULT_REWARD.map(function (n) { return build(M, n); });
      M.scripts.run = DEFAULT_RUN.map(function (n) { return build(M, n); });
    }
    function buildGuide(smooth, effort) {
      G = fresh();
      G.scripts.reward = [
        build(G, { t: "g_up" }),
        build(G, { t: "g_smooth", f: { lv: smooth } }),
        build(G, { t: "g_effort", f: { lv: effort } })
      ];
      G.scripts.run = [build(G, { t: "learn" }), build(G, { t: "test", a: [10] })];
    }

    function drop(id) {
      var b = M.B[id];
      if (!b) return;
      b.s.forEach(function (s) { if (s.block) drop(s.block); });
      b.bodies.forEach(function (list) { list.forEach(drop); });
      delete M.B[id];
    }
    function looseOf(lid) {
      for (var i = 0; i < M.loose.length; i++) if (M.loose[i].lid === lid) return M.loose[i];
      return null;
    }
    function getList(key) {
      var p = key.split(":");
      if (p[0] === "script") return M.scripts[p[1]];
      if (p[0] === "loose") { var L = looseOf(+p[1]); return L ? L.ids : []; }
      return M.B[+p[1]].bodies[+p[2]];
    }
    function pruneLoose() {
      M.loose = M.loose.filter(function (L) { return L.ids.length > 0; });
    }
    function detach(o) {
      if (!o) return;
      if (o.kind === "slot") M.B[o.parent].s[o.index].block = null;
      else getList(o.key).splice(o.index, 1);
    }
    function attach(o, id) {
      if (o.kind === "slot") {
        var s = M.B[o.parent].s[o.index];
        if (s.block) drop(s.block);
        s.block = id;
      } else {
        var list = getList(o.key);
        list.splice(Math.min(o.index, list.length), 0, id);
      }
    }

    /* ── 그리기 ── */
    function shapeSvg() {
      var s = document.createElementNS(NS, "svg");
      s.setAttribute("class", "shape");
      s.appendChild(document.createElementNS(NS, "path"));
      return s;
    }
    function paintShape(el, c) {
      el.dataset.fill = c.color;
      el.dataset.stroke = c.dark;
    }
    function sizeField(inp) {
      inp.style.width = Math.max(2, String(inp.value).length + 0.6) * 7 + 10 + "px";
    }
    function numField(slot, preview) {
      var inp = document.createElement("input");
      inp.type = "text";
      inp.inputMode = "decimal";
      inp.className = "field";
      inp.value = slot.v;
      inp.setAttribute("aria-label", "숫자");
      inp.disabled = locked || preview || mode === "guide";
      sizeField(inp);
      inp.addEventListener("input", function () {
        var x = parseFloat(inp.value);
        slot.v = isNaN(x) ? 0 : x;
        sizeField(inp);
        decorate(canvas);
        changed();
      });
      inp.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      return inp;
    }
    function ddField(b, p, c, preview) {
      var sel = document.createElement("select");
      sel.className = "dd";
      sel.style.background = c.dark;
      sel.setAttribute("aria-label", "고르기");
      p.opts.forEach(function (o) {
        var e = document.createElement("option");
        e.value = String(o[0]);
        e.textContent = o[1];
        sel.appendChild(e);
      });
      sel.value = String(b.f[p.dd]);
      sel.disabled = locked || preview;
      var fit = function () {
        var o = sel.options[sel.selectedIndex];
        sel.style.width = ((o ? o.textContent.length : 2) * 11 + 26) + "px";
      };
      fit();
      sel.addEventListener("change", function () {
        var raw = sel.value, n = Number(raw);
        b.f[p.dd] = raw !== "" && !isNaN(n) ? n : raw;
        fit();
        if (mode === "guide" && opt.onGuide) opt.onGuide(b.type, b.f[p.dd]);
        decorate(canvas);
        changed();
      });
      sel.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      var w = document.createElement("span");
      w.className = "ddw";
      w.appendChild(sel);
      return w;
    }
    function fillContent(el, b, model, preview) {
      var d = DEFS[b.type], c = PS.catOf(d.cat), k = 0;
      d.tpl.forEach(function (p) {
        if (typeof p === "string") {
          var t = document.createElement("span");
          t.className = "lbl";
          t.textContent = p;
          el.appendChild(t);
        } else if (p.s) {
          var slot = b.s[k], sp = document.createElement("span");
          sp.className = "slot " + p.s;
          sp.dataset.kind = p.s;
          if (!preview) { sp.dataset.parent = b.id; sp.dataset.index = k; }
          if (slot.block && model.B[slot.block]) sp.appendChild(blockEl(slot.block, model, preview));
          else if (p.s === "num") sp.appendChild(numField(slot, preview));
          else { var ph = document.createElement("span"); ph.className = "boolph"; sp.appendChild(ph); }
          el.appendChild(sp);
          k++;
        } else if (p.dd) {
          el.appendChild(ddField(b, p, c, preview));
        }
      });
      if (d.shape === "stk" || d.shape === "c") {
        var ic = document.createElement("span");
        ic.className = "bicon";
        ic.innerHTML = PS.iconSvg(c.key);
        el.appendChild(ic);
      }
    }
    function blockEl(id, model, preview) {
      var b = model.B[id], d = DEFS[b.type], c = PS.catOf(d.cat), el;
      if (d.shape === "val") {
        el = document.createElement("span");
        el.className = "block val";
        el.style.background = c.color;
        el.style.borderColor = c.dark;
        fillContent(el, b, model, preview);
      } else if (d.shape === "bool") {
        el = document.createElement("span");
        el.className = "block bool";
        el.dataset.shape = "bool";
        paintShape(el, c);
        el.appendChild(shapeSvg());
        fillContent(el, b, model, preview);
      } else if (d.shape === "stk") {
        el = document.createElement("div");
        el.className = "block stk";
        el.dataset.shape = "stk";
        paintShape(el, c);
        el.appendChild(shapeSvg());
        fillContent(el, b, model, preview);
        var lv = document.createElement("span");
        lv.className = "live";
        lv.hidden = true;
        el.appendChild(lv);
      } else {
        el = document.createElement("div");
        el.className = "block c";
        el.dataset.shape = "c";
        paintShape(el, c);
        el.appendChild(shapeSvg());
        var head = document.createElement("div");
        head.className = "chead";
        fillContent(head, b, model, preview);
        el.appendChild(head);
        b.bodies.forEach(function (list, i) {
          if (i > 0) {
            var mid = document.createElement("div");
            mid.className = "cmid";
            mid.textContent = d.mid || "";
            el.appendChild(mid);
          }
          var body = preview ? document.createElement("div") : listEl("body:" + b.id + ":" + i, list, d.dom, model);
          body.classList.add("cbody");
          el.appendChild(body);
        });
        var foot = document.createElement("div");
        foot.className = "cfoot";
        el.appendChild(foot);
      }
      el.dataset.id = id;
      el.dataset.type = b.type;
      if (d.tip) el.title = d.tip;
      return el;
    }
    function gapEl(key, index, dom) {
      var g = document.createElement("div");
      g.className = "gap";
      g.dataset.list = key;
      g.dataset.index = index;
      g.dataset.dom = dom;
      return g;
    }
    function listEl(key, list, dom, model) {
      var col = document.createElement("div");
      col.className = "list";
      col.dataset.list = key;
      col.dataset.dom = dom;
      col.appendChild(gapEl(key, 0, dom));
      list.forEach(function (id, i) {
        if (!model.B[id]) return;
        col.appendChild(blockEl(id, model, false));
        col.appendChild(gapEl(key, i + 1, dom));
      });
      return col;
    }
    function hatEl(name) {
      var h = HATS[name], c = PS.catOf(h.cat);
      var el = document.createElement("div");
      el.className = "hatb";
      el.dataset.shape = "hat";
      el.dataset.hat = name;
      paintShape(el, c);
      el.appendChild(shapeSvg());
      var ic = document.createElement("span");
      ic.className = "hicon";
      ic.style.color = c.color;
      ic.innerHTML = PS.iconSvg(c.key);
      el.appendChild(ic);
      var t = document.createElement("span");
      t.className = "lbl";
      t.textContent = h.label;
      el.appendChild(t);
      el.title = h.tip;
      return el;
    }
    function scriptEl(name, model) {
      var wrap = document.createElement("div");
      wrap.className = "script";
      wrap.appendChild(hatEl(name));
      var under = listEl("script:" + name, model.scripts[name], mode === "guide" ? "guide" : name, model);
      under.classList.add("under");
      if (!model.scripts[name].length) {
        var e = document.createElement("div");
        e.className = "empty";
        e.textContent = "여기에 블록을 끌어다 붙이세요";
        under.appendChild(e);
      }
      wrap.appendChild(under);
      return wrap;
    }

    function pathFor(el) {
      var kind = el.dataset.shape, W = el.offsetWidth, H = el.offsetHeight, r, d;
      if (kind === "stk") {
        r = H / 2;
        d = "M0 0 l6 6 l6 -6 H" + (W - r) + " a" + r + " " + r + " 0 0 1 0 " + H + " H12 l-6 6 l-6 -6 z";
        return { d: d, w: W, h: H + 7 };
      }
      if (kind === "hat") {
        d = "M24 34 H" + (W - 14) + " a14 14 0 0 0 0 -28 H30 A17 17 0 0 0 18 1 C8.611 1 1 8.611 1 18 c0 7.2 4.5 13.5 11 16 l6 6 l6 -6 z";
        return { d: d, w: W, h: 41 };
      }
      if (kind === "bool") {
        var e = 0.4 * H;
        d = "M" + e + " 0 L0 " + (H / 2) + " L" + e + " " + H + " H" + (W - e) + " L" + W + " " + (H / 2) + " L" + (W - e) + " 0 z";
        return { d: d, w: W, h: H };
      }
      var head = el.querySelector(":scope > .chead");
      var bodies = el.querySelectorAll(":scope > .cbody"), mids = el.querySelectorAll(":scope > .cmid");
      var hw = head.offsetWidth, hh = head.offsetHeight;
      r = hh / 2;
      d = "M0 0 l6 6 l6 -6 H" + (hw - r) + " a" + r + " " + r + " 0 0 1 0 " + hh + " H26 l-6 6 l-6 -6";
      for (var i = 0; i < bodies.length; i++) {
        d += " v" + bodies[i].offsetHeight + " l6 6 l6 -6";
        if (i < bodies.length - 1) {
          var mh = mids[i] ? mids[i].offsetHeight : 28;
          d += " H" + (hw - 22) + " a" + (mh / 2) + " " + (mh / 2) + " 0 0 1 0 " + mh + " H26 l-6 6 l-6 -6";
        }
      }
      d += " H" + (hw - 22) + " a7.5 7.5 0 0 1 0 15 H12 l-6 6 l-6 -6 z";
      return { d: d, w: hw, h: H + 7 };
    }
    function decorate(root) {
      var list = root.querySelectorAll("[data-shape]");
      for (var i = 0; i < list.length; i++) {
        var el = list[i], svg = el.firstElementChild;
        if (!svg || !el.offsetWidth) continue;
        var p = pathFor(el), path = svg.firstChild;
        path.setAttribute("d", p.d);
        path.setAttribute("fill", el.dataset.fill);
        path.setAttribute("stroke", el.dataset.stroke);
        svg.setAttribute("width", p.w + 2);
        svg.setAttribute("height", p.h + 2);
      }
    }

    /* 따로 놓아 둔 묶음. 쌓기 블록이면 사이에 끼울 수 있는 목록, 값·판단 블록이면 블록 하나. */
    function looseEl(L, model) {
      var wrap = document.createElement("div");
      wrap.className = "script loose";
      wrap.title = "모자 블록에 붙어 있지 않아 실행되지 않습니다";
      var first = L.ids.length ? DEFS[model.B[L.ids[0]].type] : null, col;
      if (first && (first.shape === "val" || first.shape === "bool")) {
        col = document.createElement("div");
        col.className = "list";
        col.dataset.list = "loose:" + L.lid;
        col.appendChild(blockEl(L.ids[0], model, false));
      } else {
        col = listEl("loose:" + L.lid, L.ids, first ? first.dom : "*", model);
      }
      wrap.appendChild(col);
      return wrap;
    }
    function place(el, x, y) {
      el.style.left = x + "px";
      el.style.top = y + "px";
    }
    function fitCanvas() {
      var w = 0, h = 0, kids = canvas.children;
      for (var i = 0; i < kids.length; i++) {
        w = Math.max(w, kids[i].offsetLeft + kids[i].offsetWidth);
        h = Math.max(h, kids[i].offsetTop + kids[i].offsetHeight);
      }
      canvas.style.width = (w + 80) + "px";
      canvas.style.height = (h + 90) + "px";
    }
    function render() {
      var model = cur(), pos = model.pos || {};
      canvas.innerHTML = "";
      var a = scriptEl("reward", model);
      canvas.appendChild(a);
      var pa = pos.reward || { x: 26, y: 22 };
      place(a, pa.x, pa.y);
      var b = scriptEl("run", model);
      canvas.appendChild(b);
      var pb = pos.run || { x: pa.x, y: pa.y + a.offsetHeight + 34 };
      place(b, pb.x, pb.y);
      (model.loose || []).forEach(function (L) {
        var e = looseEl(L, model);
        canvas.appendChild(e);
        place(e, L.x, L.y);
      });
      board.classList.toggle("guided", mode === "guide");
      fitCanvas();
      decorate(canvas);
    }

    /* ── 꾸러미(팔레트) ── */
    function paint() {
      paletteEl.innerHTML = "";
      var c = PS.catOf(cat);
      railEl.querySelectorAll("button").forEach(function (b) {
        var on = b.dataset.cat === cat, cc = PS.catOf(b.dataset.cat);
        b.setAttribute("aria-pressed", String(on));
        b.style.background = on ? cc.color : "";
        b.querySelector("i").style.background = on ? "#fff" : cc.color;
        b.querySelector("i").style.color = on ? cc.color : "#fff";
      });
      if (mode === "guide") {
        var n = document.createElement("p");
        n.className = "palnote";
        n.textContent = "따라 하기에서는 블록이 정해져 있어요. 블록 안의 고르는 칸(▼)만 바꿔 보세요. " +
          "블록을 직접 쌓으려면 위에서 '자유롭게 만들기'를 누르세요.";
        paletteEl.appendChild(n);
        return;
      }
      var tmp = fresh(), lastGroup = null;
      if (cat === "start") {
        var note = document.createElement("p");
        note.className = "palnote";
        note.textContent = "시작 블록 두 개는 조립소에 이미 놓여 있습니다. 그 아래에 다른 블록을 붙이세요.";
        paletteEl.appendChild(note);
        ["reward", "run"].forEach(function (name) {
          var row = document.createElement("div");
          row.className = "palrow hatrow";
          row.appendChild(hatEl(name));
          paletteEl.appendChild(row);
        });
      }
      PALETTE[cat].forEach(function (type) {
        var d = DEFS[type];
        var group = d.group || c.name + " · " + c.note;
        if (group !== lastGroup) {
          var h = document.createElement("h3");
          h.textContent = group;
          paletteEl.appendChild(h);
          lastGroup = group;
        }
        var b = mkIn(tmp, type);
        var row = document.createElement("div");
        row.className = "palrow";
        var el = blockEl(b.id, tmp, true);
        el.classList.add("palblock");
        el.dataset.new = type;
        row.appendChild(el);
        paletteEl.appendChild(row);
      });
      var r = document.createElement("button");
      r.className = "reset";
      r.textContent = "처음 블록으로 되돌리기";
      r.disabled = locked;
      r.addEventListener("click", function () { defaults(); render(); changed(); });
      paletteEl.appendChild(r);
      decorate(paletteEl);
    }

    railEl.innerHTML = "";
    CATS.forEach(function (c) {
      var b = document.createElement("button");
      b.dataset.cat = c.key;
      b.innerHTML = "<i></i><span></span>";
      b.querySelector("i").innerHTML = PS.iconSvg(c.key);
      b.querySelector("span").textContent = c.name;
      b.addEventListener("click", function () { cat = c.key; paint(); });
      railEl.appendChild(b);
    });

    /* ── 끌어 놓기 ──
     * 쌓기 블록을 끌면 엔트리처럼 그 아래에 붙은 블록도 함께 움직입니다(drag.ids). */
    function originOf(el) {
      var id = +el.dataset.id, p = el.parentNode;
      if (p && p.classList.contains("slot")) return { kind: "slot", parent: +p.dataset.parent, index: +p.dataset.index, id: id };
      if (p && p.classList.contains("list")) {
        var list = getList(p.dataset.list);
        return { kind: "list", key: p.dataset.list, index: list.indexOf(id), id: id };
      }
      return null;
    }
    function ghostFrom(ids) {
      ghost.innerHTML = "";
      var col = document.createElement("div");
      col.className = "list";
      ids.forEach(function (id) { col.appendChild(blockEl(id, M, false)); });
      ghost.appendChild(col);
      ghost.hidden = false;
      ghost.style.transform = "scale(" + zoom + ")";
      decorate(ghost);
    }
    function clearTargets() {
      board.querySelectorAll(".over").forEach(function (e) { e.classList.remove("over"); });
    }
    function findSlot(x, y, kind) {
      var list = document.elementsFromPoint(x, y);
      for (var i = 0; i < list.length; i++) {
        var e = list[i];
        if (ghost.contains(e) || !canvas.contains(e)) continue;
        if (e.classList && e.classList.contains("slot") && e.dataset.kind === kind && e.dataset.parent) return e;
      }
      return null;
    }
    function findGap(x, y, dom) {
      var gaps = canvas.querySelectorAll(".gap"), best = null, bestScore = 1e9;
      for (var i = 0; i < gaps.length; i++) {
        var g = gaps[i];
        if (g.dataset.dom !== dom && g.dataset.dom !== "*") continue;
        var r = g.getBoundingClientRect();
        var dy = Math.abs(y - r.top), dx = x - r.left;
        if (dy > 22 * zoom || dx < -40 * zoom || dx > 300 * zoom) continue;
        var score = dy - r.left * 0.001;
        if (score < bestScore) { bestScore = score; best = g; }
      }
      return best;
    }
    function targetFor(x, y) {
      if (!drag) return null;
      if (drag.shape === "val") return findSlot(x, y, "num");
      if (drag.shape === "bool") return findSlot(x, y, "bool");
      return findGap(x, y, drag.dom);
    }
    function pick(type, hat) {
      if (!opt.onPick) return;
      if (hat) { opt.onPick(HATS[hat].label, HATS[hat].tip); return; }
      var d = DEFS[type];
      var tmp = fresh(), b = mkIn(tmp, type);
      opt.onPick(PS.reward.blockText(tmp, b.id), d.tip || "");
    }
    function insertAt(key, index, ids) {
      var list = getList(key);
      Array.prototype.splice.apply(list, [Math.min(index, list.length), 0].concat(ids));
    }
    function putBack() {
      if (!drag) return;
      if (!drag.origin) { drag.ids.forEach(drop); return; }
      if (drag.origin.kind === "slot") attach(drag.origin, drag.ids[0]);
      else insertAt(drag.origin.key, drag.origin.index, drag.ids);
    }

    board.addEventListener("pointerdown", function (e) {
      if (e.target.closest(".zoomer")) return;
      var hat = e.target.closest(".hatb");
      if (hat) {
        pick(null, hat.dataset.hat);
        if (locked || mode === "guide") return;
        var sc = hat.parentNode, hr = sc.getBoundingClientRect();
        drag = { script: hat.dataset.hat, el: sc, dx: e.clientX - hr.left, dy: e.clientY - hr.top };
        sc.classList.add("moving");
        board.setPointerCapture(e.pointerId);
        e.preventDefault();
        return;
      }
      var el = e.target.closest(".block");
      if (!el) return;
      pick(el.dataset.type);
      if (locked || mode === "guide") return;
      var o = originOf(el);
      if (!o) return;
      var d = DEFS[M.B[o.id].type], r = el.getBoundingClientRect(), ids;
      if (o.kind === "list") ids = getList(o.key).splice(o.index);
      else { ids = [o.id]; detach(o); }
      drag = { ids: ids, origin: o, shape: d.shape, dom: d.dom, dx: e.clientX - r.left, dy: e.clientY - r.top };
      ghostFrom(ids);
      render();
      moveGhost(e.clientX, e.clientY);
      board.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    paletteEl.addEventListener("pointerdown", function (e) {
      var hat = e.target.closest(".hatb");
      if (hat) { pick(null, hat.dataset.hat); return; }
      var el = e.target.closest(".palblock");
      if (!el) return;
      pick(el.dataset.new);
      if (locked || mode === "guide") return;
      var type = el.dataset.new, d = DEFS[type];
      var b = mk(type), r = el.getBoundingClientRect();
      drag = { ids: [b.id], origin: null, shape: d.shape, dom: d.dom, dx: (e.clientX - r.left) * zoom, dy: (e.clientY - r.top) * zoom };
      ghostFrom(drag.ids);
      moveGhost(e.clientX, e.clientY);
      paletteEl.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    function moveGhost(x, y) {
      var br = board.getBoundingClientRect();
      ghost.style.left = (x - br.left + board.scrollLeft - drag.dx) + "px";
      ghost.style.top = (y - br.top + board.scrollTop - drag.dy) + "px";
    }
    function overTrash(x, y) {
      var r = trash.getBoundingClientRect();
      return x >= r.left - 10 && x <= r.right + 10 && y >= r.top - 10 && y <= r.bottom + 10;
    }
    function boardXY(x, y) {
      var br = board.getBoundingClientRect();
      return {
        x: Math.max(0, Math.round((x - br.left + board.scrollLeft - drag.dx) / zoom)),
        y: Math.max(0, Math.round((y - br.top + board.scrollTop - drag.dy) / zoom))
      };
    }
    function insideBoard(x, y) {
      var br = board.getBoundingClientRect();
      return x >= br.left && x <= br.right && y >= br.top && y <= br.bottom;
    }
    function onMove(e) {
      if (!drag) return;
      if (drag.script) {
        var q = boardXY(e.clientX, e.clientY);
        place(drag.el, q.x, q.y);
        return;
      }
      moveGhost(e.clientX, e.clientY);
      clearTargets();
      var t = targetFor(e.clientX, e.clientY);
      if (t) t.classList.add("over");
      trash.classList.toggle("over", overTrash(e.clientX, e.clientY));
    }
    function onUp(e) {
      if (!drag) return;
      if (drag.script) {
        M.pos[drag.script] = boardXY(e.clientX, e.clientY);
        drag = null;
        render();
        changed();
        return;
      }
      var t = targetFor(e.clientX, e.clientY);
      var hit = document.elementFromPoint(e.clientX, e.clientY);
      var away = overTrash(e.clientX, e.clientY) || (hit && paletteEl.contains(hit));
      clearTargets();
      trash.classList.remove("over");
      ghost.hidden = true;
      ghost.innerHTML = "";
      if (away) {
        drag.ids.forEach(drop);
      } else if (t && t.classList.contains("slot")) {
        attach({ kind: "slot", parent: +t.dataset.parent, index: +t.dataset.index }, drag.ids[0]);
      } else if (t) {
        insertAt(t.dataset.list, +t.dataset.index, drag.ids);
      } else if (insideBoard(e.clientX, e.clientY)) {
        /* 조립소 빈 곳에 놓으면 그 자리에 따로 놓아 둡니다(엔트리와 같음). */
        var q = boardXY(e.clientX, e.clientY);
        M.loose.push({ lid: M.next++, x: q.x, y: q.y, ids: drag.ids });
      } else if (drag.origin) {
        putBack();
      } else {
        drag.ids.forEach(drop);
      }
      drag = null;
      pruneLoose();
      render();
      changed();
    }
    board.addEventListener("pointermove", onMove);
    paletteEl.addEventListener("pointermove", onMove);
    board.addEventListener("pointerup", onUp);
    paletteEl.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", function () {
      if (!drag) return;
      ghost.hidden = true;
      if (!drag.script) putBack();
      drag = null;
      pruneLoose();
      clearTargets();
      render();
    });

    /* ── 되돌리기 / 다시 실행 ── */
    function commit() {
      var now = JSON.stringify(M);
      if (hist[hpos] === now) return;
      hist = hist.slice(0, hpos + 1);
      hist.push(now);
      if (hist.length > 60) hist.shift();
      hpos = hist.length - 1;
    }
    function restore(i) {
      if (i < 0 || i >= hist.length || locked || mode === "guide") return false;
      hpos = i;
      M = JSON.parse(hist[i]);
      render();
      paint();
      if (opt.onChange) opt.onChange();
      return true;
    }

    /* ── 조립소 확대·축소 ── */
    function setZoom(z) {
      zoom = Math.max(0.6, Math.min(2, Math.round(z * 100) / 100));
      canvas.style.transform = "scale(" + zoom + ")";
      zoomer.querySelector("b").textContent = Math.round(zoom * 100) + "%";
      try { localStorage.setItem("ps.zoom", String(zoom)); } catch (err) { return; }
    }
    zoomer.addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.z === "out") setZoom(zoom - 0.1);
      else if (b.dataset.z === "in") setZoom(zoom + 0.1);
      else setZoom(1);
    });

    /* ── 바깥에서 쓰는 것 ── */
    self.model = function () { return M; };
    self.mode = function () { return mode; };
    self.setMode = function (m, g) {
      mode = m;
      if (m === "guide") buildGuide(g.smooth, g.effort);
      render();
      paint();
    };
    self.refresh = function () { render(); paint(); };
    self.lock = function (on) { locked = on; board.classList.toggle("locked", on); render(); paint(); };
    self.rule = function () { return PS.reward.blocks(M); };
    self.runOpts = function () { return PS.reward.runOpts(cur()); };
    self.reset = function () { defaults(); render(); paint(); changed(); };
    self.undo = function () { return restore(hpos - 1); };
    self.redo = function () { return restore(hpos + 1); };
    self.canUndo = function () { return mode === "free" && hpos > 0; };
    self.canRedo = function () { return mode === "free" && hpos < hist.length - 1; };
    self.mark = function (type) {
      canvas.querySelectorAll(".block.running").forEach(function (e) { e.classList.remove("running"); });
      if (!type) return;
      var el = canvas.querySelector('.list[data-list="script:run"] > .block[data-type="' + type + '"]');
      if (el) el.classList.add("running");
    };
    self.loadReward = function (nodes) {
      M.scripts.reward.slice().forEach(drop);
      M.scripts.reward = nodes.map(function (n) { return build(M, n); });
      render();
      changed();
    };
    self.exportModel = function () { return JSON.parse(JSON.stringify(M)); };
    self.importModel = function (o) {
      if (!o || !o.B || !o.scripts || !o.scripts.reward || !o.scripts.run) return false;
      for (var k in o.B) if (!DEFS[o.B[k].type]) return false;
      M = JSON.parse(JSON.stringify(o));
      if (!M.pos) M.pos = { reward: null, run: null };
      if (!M.loose) M.loose = [];
      render();
      changed();
      return true;
    };
    self.setCheck = function (c) {
      checkEl.className = "rulecheck " + (c ? c.level : "");
      checkEl.textContent = c ? (c.level === "ok" ? "규칙 점검 ✔ " : c.level === "warn" ? "규칙 점검 ⚠ " : "") + c.msg : "";
      checkEl.hidden = !c;
    };
    self.live = function (map) {
      var els = canvas.querySelectorAll(".live");
      for (var i = 0; i < els.length; i++) {
        var el = els[i], x = map ? map[+el.parentNode.dataset.id] : undefined;
        if (x === undefined) { el.hidden = true; continue; }
        var a = Math.abs(x);
        el.textContent = (x < 0 ? "−" : "+") + (a >= 10 ? a.toFixed(0) : a.toFixed(1));
        el.className = "live" + (x < 0 ? " neg" : "");
        el.hidden = false;
      }
    };

    defaults();
    buildGuide(1, 1);
    commit();
    try { setZoom(parseFloat(localStorage.getItem("ps.zoom")) || 1); } catch (err) { setZoom(1); }
    render();
    paint();
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { decorate(canvas); decorate(paletteEl); });
    }
  }

  PS.Blocks = Blocks;
})(window.PS = window.PS || {});
