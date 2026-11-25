/* 블록 정의.
 *
 * 범주 이름·색·순서와 블록 모양은 엔트리를 따릅니다 (app/design/블록_UI_개선_설계안.md).
 *   stk  쌓기 블록 — 왼쪽 위 V 홈, 왼쪽 아래 V 돌기, 오른쪽 끝 반원
 *   c    감싸는 블록 — 만일 ~ (이)라면
 *   val  값 블록 — 둥근 캡슐. 숫자 칸에 끼웁니다
 *   bool 판단 블록 — 육각. 육각 칸에만 끼웁니다
 *
 * tpl 은 블록에 적히는 글과 칸이 번갈아 오는 목록입니다.
 *   "글"                         그대로 적는 글
 *   { s: "num", v: 10 }          숫자 칸 (값 블록을 끼울 수 있음), v 는 처음 값
 *   { s: "bool" }                육각 칸 (판단 블록만)
 *   { dd: "이름", opts: [[값, 글], ...] }   고르는 칸
 *
 * dom 은 쌓기 블록이 들어갈 수 있는 묶음입니다. reward = 점수 규칙, run = 시작하기 아래.
 *
 * 키트 값은 모두 0~100 으로 맞췄습니다. 방향(부호)이 없어서 제곱·절댓값 없이도 "빠를수록 감점"을 쓸 수 있고,
 * 블록끼리 크기를 견줄 수 있습니다. v = { phi, ald, thd, u, th } (phi 는 똑바로 섰을 때 0).
 */
(function (PS) {
  "use strict";

  var PEND_SPEED_MAX = 25.0;
  var ARM_SPEED_MAX = 19.2;
  var DEG = 180 / Math.PI;

  var CATS = [
    { key: "start", name: "시작", color: "#00b400", dark: "#009400", glyph: "▶", note: "프로그램이 시작되는 곳" },
    { key: "flow", name: "흐름", color: "#19baea", dark: "#1498c0", glyph: "⤴", note: "조건에 따라 실행하기" },
    { key: "judge", name: "판단", color: "#4562f5", dark: "#1b3ad8", glyph: "✓", note: "참인지 거짓인지" },
    { key: "calc", name: "계산", color: "#f4af18", dark: "#ff7f00", glyph: "＋", note: "숫자 계산" },
    { key: "score", name: "점수", color: "#dd47d8", dark: "#b819b3", glyph: "★", note: "보상(점수) 규칙" },
    { key: "kit", name: "키트", color: "#00b6b1", dark: "#008f8b", glyph: "⚙", note: "키트가 재는 값" },
    { key: "ai", name: "인공지능", color: "#8222ff", dark: "#6100d6", glyph: "AI", note: "가르치고 시험하기" }
  ];

  function clamp100(x) { return x < 0 ? 0 : x > 100 ? 100 : x; }
  var LEVELS = [[0, "감점 없음"], [1, "약하게 감점"], [2, "강하게 감점"]];

  var DEFS = {
    /* ── 키트: 값 (0~100) ── */
    height: { cat: "kit", shape: "val", tpl: ["진자 높이 값"], group: "키트가 재는 값 (0~100)",
      tip: "진자가 아래로 매달리면 0, 똑바로 서면 100 입니다.",
      read: function (v) { return 50 * (1 + Math.cos(v.phi)); } },
    pvel: { cat: "kit", shape: "val", tpl: ["진자 빠르기 값"], group: "키트가 재는 값 (0~100)",
      tip: "진자가 멈춰 있으면 0, 안전 정지 속도로 돌면 100 입니다. 방향은 따지지 않습니다.",
      read: function (v) { return clamp100(Math.abs(v.ald) / PEND_SPEED_MAX * 100); } },
    avel: { cat: "kit", shape: "val", tpl: ["팔 빠르기 값"], group: "키트가 재는 값 (0~100)",
      tip: "팔이 멈춰 있으면 0, 안전 정지 속도로 돌면 100 입니다. 방향은 따지지 않습니다.",
      read: function (v) { return clamp100(Math.abs(v.thd) / ARM_SPEED_MAX * 100); } },
    force: { cat: "kit", shape: "val", tpl: ["모터 힘 값"], group: "키트가 재는 값 (0~100)",
      tip: "모터가 쉬면 0, 가장 세게 밀면 100 입니다. 방향은 따지지 않습니다.",
      read: function (v) { return clamp100(Math.abs(v.u) * 100); } },
    tilt: { cat: "kit", shape: "val", tpl: ["진자 기울기 값"], group: "각도 (도)",
      tip: "똑바로 서면 0도, 아래로 매달리면 180도입니다.",
      read: function (v) { return Math.abs(v.phi) * DEG; } },
    armang: { cat: "kit", shape: "val", tpl: ["팔 각도 값"], group: "각도 (도)",
      tip: "팔이 처음 자리에서 돌아간 각도(도)입니다. 한 바퀴 돌면 360 입니다.",
      read: function (v) { return Math.abs(v.th) * DEG; } },
    upright: { cat: "kit", shape: "bool", tpl: ["진자가 똑바로 섰는가?"], group: "판단",
      tip: "기울기가 10도 안이고 천천히 움직이면 참입니다.",
      test: function (v) { return Math.abs(v.phi) < 0.1745 && Math.abs(v.ald) < 2; } },
    home: { cat: "kit", shape: "bool", tpl: ["팔이 제자리에 있는가?"], group: "판단",
      tip: "팔 각도가 처음 자리에서 30도 안이면 참입니다.",
      test: function (v) { return Math.abs(v.th) * DEG < 30; } },

    /* ── 계산 ── */
    add: { cat: "calc", shape: "val", tpl: [{ s: "num", v: 10 }, "+", { s: "num", v: 10 }], fn: function (a, b) { return a + b; } },
    sub: { cat: "calc", shape: "val", tpl: [{ s: "num", v: 10 }, "-", { s: "num", v: 10 }], fn: function (a, b) { return a - b; } },
    mul: { cat: "calc", shape: "val", tpl: [{ s: "num", v: 10 }, "x", { s: "num", v: 0.5 }], fn: function (a, b) { return a * b; },
      tip: "값에 숫자를 곱해 크기를 조절합니다. 0.5 를 곱하면 절반이 됩니다." },
    div: { cat: "calc", shape: "val", tpl: [{ s: "num", v: 10 }, "/", { s: "num", v: 2 }],
      fn: function (a, b) { return b === 0 ? 0 : a / b; } },
    pow: { cat: "calc", shape: "val", tpl: [{ s: "num", v: 10 }, "의", { dd: "op", opts: [["sq", "제곱"], ["abs", "절댓값"]] }],
      fn: function (a, f) { return f.op === "abs" ? Math.abs(a) : a * a; } },

    /* ── 판단 ── */
    lt: { cat: "judge", shape: "bool", tpl: [{ s: "num", v: 0 }, "<", { s: "num", v: 10 }], fn: function (a, b) { return a < b; },
      tip: "왼쪽 값이 오른쪽 값보다 작으면 참입니다." },
    gt: { cat: "judge", shape: "bool", tpl: [{ s: "num", v: 0 }, ">", { s: "num", v: 10 }], fn: function (a, b) { return a > b; },
      tip: "왼쪽 값이 오른쪽 값보다 크면 참입니다." },
    and: { cat: "judge", shape: "bool", tpl: [{ s: "bool" }, "그리고", { s: "bool" }], fnb: function (a, b) { return a && b; } },
    or: { cat: "judge", shape: "bool", tpl: [{ s: "bool" }, "또는", { s: "bool" }], fnb: function (a, b) { return a || b; } },
    not: { cat: "judge", shape: "bool", tpl: [{ s: "bool" }, "(이)가 아니다"], fnb: function (a) { return !a; } },

    /* ── 흐름 ── */
    "if": { cat: "flow", shape: "c", dom: "reward", bodies: 1, tpl: ["만일", { s: "bool" }, "(이)라면"],
      tip: "육각 칸이 참일 때만 안쪽 블록을 실행합니다." },
    ifelse: { cat: "flow", shape: "c", dom: "reward", bodies: 2, mid: "아니면", tpl: ["만일", { s: "bool" }, "(이)라면"],
      tip: "참이면 위쪽, 거짓이면 아래쪽 블록을 실행합니다." },

    /* ── 점수 ── */
    plus: { cat: "score", shape: "stk", dom: "reward", tpl: ["점수에", { s: "num", v: 10 }, "만큼 더하기"],
      tip: "이 순간의 점수를 올립니다. 에이전트는 점수가 커지는 쪽으로 배웁니다." },
    minus: { cat: "score", shape: "stk", dom: "reward", tpl: ["점수에서", { s: "num", v: 10 }, "만큼 빼기"],
      tip: "이 순간의 점수를 깎습니다. 하지 않았으면 하는 행동에 씁니다." },

    /* ── 따라 하기 (고르기만 하는 블록, 미리 학습한 아홉 정책과 짝) ── */
    g_up: { cat: "score", shape: "stk", dom: "guide", tpl: ["진자가 설수록 점수 주기"],
      tip: "항상 켜져 있습니다. 거꾸로 서 있을수록 점수를 줍니다." },
    g_smooth: { cat: "score", shape: "stk", dom: "guide", tpl: ["진자를 빠르게 휘두르면", { dd: "lv", opts: LEVELS }],
      tip: "부드럽게: 빠르게 휘두르면 점수를 깎습니다." },
    g_effort: { cat: "score", shape: "stk", dom: "guide", tpl: ["모터를 세게 쓰면", { dd: "lv", opts: LEVELS }],
      tip: "힘 아끼기: 모터를 세게 쓰면 점수를 깎습니다." },

    /* ── 인공지능 ── */
    cond: { cat: "ai", shape: "stk", dom: "run",
      tpl: ["학습 조건", { dd: "key", opts: [["gens", "세대"], ["pop", "한 세대에 해 볼 수"], ["sigma", "바꾸는 크기"]] },
        "을(를)", { s: "num", v: 24 }, "(으)로 바꾸기"],
      tip: "세대: 몇 번 고쳐 볼지 (4~80). 한 세대에 해 볼 수: 한 번에 시험하는 정책 수 (6~40). 바꾸는 크기: 얼마나 과감하게 바꿀지 (1~30)." },
    learn: { cat: "ai", shape: "stk", dom: "run", tpl: ["내 점수 규칙으로 정책 학습하기"],
      tip: "미리 배운 정책에서 출발해 내 점수 규칙에 맞게 고쳐 배웁니다." },
    test: { cat: "ai", shape: "stk", dom: "run", tpl: [{ s: "num", v: 10 }, "초 동안 시험하기"],
      tip: "배운 정책으로 진자를 움직여 봅니다 (3~20초). 시뮬레이션인지 실제 키트인지는 시작하기 옆에서 고릅니다." }
  };

  var HATS = {
    reward: { cat: "score", glyph: "★", label: "매 순간 점수 매기기",
      tip: "키트가 0.02초마다 한 번씩 이 아래 블록을 실행해 점수를 매깁니다." },
    run: { cat: "start", glyph: "▶", label: "시작하기 버튼을 클릭했을 때",
      tip: "시작하기를 누르면 이 아래 블록을 위에서부터 실행합니다." }
  };

  var PALETTE = {
    start: [],
    flow: ["if", "ifelse"],
    judge: ["lt", "gt", "and", "or", "not"],
    calc: ["add", "sub", "mul", "div", "pow"],
    score: ["plus", "minus"],
    kit: ["height", "pvel", "avel", "force", "tilt", "armang", "upright", "home"],
    ai: ["cond", "learn", "test"]
  };

  /* 범주 아이콘 (20x20, 색은 currentColor). 글자 기호는 글꼴에 따라 깨지므로 그림으로 그립니다. */
  var ICONS = {
    start: '<path d="M7 4.2v11.6l9.4-5.8z" fill="currentColor"/>',
    flow: '<path d="M14.6 6.6A5.6 5.6 0 1 0 15.6 11" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round"/><path d="M16.4 2.8v4.6h-4.6z" fill="currentColor"/>',
    judge: '<path d="M4.4 10.6l3.6 3.6 7.6-8.2" fill="none" stroke="currentColor" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>',
    calc: '<path d="M10 4.4v11.2M4.4 10h11.2" fill="none" stroke="currentColor" stroke-width="2.7" stroke-linecap="round"/>',
    score: '<path d="M10 2.6l2.3 4.7 5.1.7-3.7 3.6.9 5.1-4.6-2.4-4.6 2.4.9-5.1L2.6 8l5.1-.7z" fill="currentColor"/>',
    kit: '<rect x="4" y="14.2" width="12" height="3" rx="1.5" fill="currentColor"/><path d="M10 14V7" stroke="currentColor" stroke-width="2.3" stroke-linecap="round"/><circle cx="10" cy="4.9" r="2.7" fill="currentColor"/>',
    ai: '<text x="10" y="13.8" font-size="9.5" font-weight="800" text-anchor="middle" fill="currentColor" font-family="Arial, sans-serif">AI</text>'
  };
  function iconSvg(key) {
    return '<svg viewBox="0 0 20 20" width="100%" height="100%" aria-hidden="true">' + (ICONS[key] || "") + "</svg>";
  }

  function catOf(key) {
    for (var i = 0; i < CATS.length; i++) if (CATS[i].key === key) return CATS[i];
    return CATS[0];
  }

  PS.CATS = CATS;
  PS.DEFS = DEFS;
  PS.HATS = HATS;
  PS.PALETTE = PALETTE;
  PS.catOf = catOf;
  PS.iconSvg = iconSvg;
  PS.blockColor = function (type) { return catOf(DEFS[type].cat).color; };
})(window.PS = window.PS || {});
