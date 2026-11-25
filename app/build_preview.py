"""앱 소스에서 웹 미리보기 페이지를 만듭니다.

Electron 앱은 창 전체를 쓰지만 미리보기는 페이지 안의 한 상자에 들어가야 하므로,
body 에 걸려 있던 레이아웃 규칙을 .appwin 으로 옮기고 바깥 껍데기를 붙입니다.
앱 소스는 하나만 유지하고 여기서 기계적으로 변환합니다.

    python app/build_preview.py
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "preview")

SHELL_HEAD = """<meta name="claude:viewport-fit-sides" content="cover">
<title>Pendulum Studio</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="css/app.css">
<style>
:root{
  --page-bg:#F4F6F8; --page-fg:#16202A; --page-dim:#5A6775; --page-line:#DCE2E8; --page-accent:#0B5C86;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --page-bg:#13181D; --page-fg:#E8EDF2; --page-dim:#9AA7B4; --page-line:#2A333C; --page-accent:#6FB6DC; color-scheme: dark; } }
:root[data-theme="dark"]{
  --page-bg:#13181D; --page-fg:#E8EDF2; --page-dim:#9AA7B4; --page-line:#2A333C; --page-accent:#6FB6DC; color-scheme: dark; }
body{
  background:var(--page-bg); color:var(--page-fg); margin:0;
  font-family:'Pretendard Variable',Pretendard,Inter,'Noto Sans KR','Malgun Gothic',system-ui,sans-serif;
  padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px));
  padding-block:22px 28px;
}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.lede h1{margin:0 0 6px;font-size:clamp(20px,3vw,27px);letter-spacing:-.02em;text-wrap:balance}
.lede p{margin:0;color:var(--page-dim);font-size:14.5px;line-height:1.6;max-width:62ch}
.lede b{color:var(--page-fg);font-weight:600}
.tipbar{display:flex;flex-wrap:wrap;gap:8px;font-size:12.5px;color:var(--page-dim)}
.tipbar span{border:1px solid var(--page-line);border-radius:99px;padding:3px 11px}
.tipbar span b{color:var(--page-accent);font-weight:600}
.frame{
  overflow-x:auto;overflow-y:hidden;border:1px solid var(--page-line);border-radius:12px;
  background:#EDF1F0;box-shadow:0 10px 30px -18px rgba(0,0,0,.45);
}
.appwin{min-width:1120px;height:770px}
.appwin .modal-back{position:absolute;inset:0}
.frame{position:relative}
.foot{color:var(--page-dim);font-size:12.5px;line-height:1.7}
.foot code{background:rgba(127,127,127,.16);padding:1px 5px;border-radius:4px;font-size:12px}
</style>
"""

LEDE = """<div class="wrap">
<div class="lede">
<h1>Pendulum Studio</h1>
<p>교육용 회전 도립 진자 키트로 강화학습을 가르치는 Electron 앱의 실제 화면입니다.
화면 배치와 블록 모양은 <b>엔트리</b>를 따르고, 활동 순서는 학습지와 짝을 이룹니다.
진자는 검증된 MuJoCo 모델에서 뽑은 동역학으로 움직이고, 정책은 실제로 학습한 신경망 가중치입니다.</p>
</div>
<div class="tipbar">
<span><b>살펴보기</b> 미리 배운 에이전트를 실행해 보세요</span>
<span><b>블록</b> 탭에서 '자유롭게 만들기'를 누르고 블록을 끌어다 쌓아 보세요</span>
<span><b>▶ 시작하기</b> 를 누르면 배우고 나서 시험합니다</span>
<span><b>예시 규칙</b> 탭에서 완성된 규칙을 불러올 수 있습니다</span>
<span>규칙을 바꾸면 <b>움직임이 실제로 달라집니다</b></span>
</div>
<div class="frame"><div class="appwin">
"""

FOOT = """</div></div>
<p class="foot">웹 미리보기라 시리얼 포트가 없어 실제 키트 연결은 비활성입니다. 데스크톱 앱에서는
<code>main.js</code>가 포트를 직접 잡습니다. 기본 규칙 아홉 조합은 모두 MuJoCo에서 따로 학습한
정책이 들어 있고, 지연·잡음을 랜덤화해 학습했으므로 관측에 최근 명령 두 개가 포함됩니다(6 → 8개).
<b>빠르기 강하게 + 힘 아끼기</b> 두 조합은 일부러 못 세웁니다 &mdash; 진자 속도를 세게 깎으면
스윙업에 필요한 에너지를 넣을 수 없기 때문이고, 학습지가 기록하게 하는 결과가 바로 이것입니다.</p>
</div>
"""


def main():
    os.makedirs(os.path.join(OUT, "css"), exist_ok=True)
    html = io.open(os.path.join(HERE, "src", "index.html"), encoding="utf-8").read()
    body = html[html.index("<body>") + len("<body>"):html.index("</body>")]
    body = body.replace('<link rel="stylesheet" href="css/app.css">', "")
    page = SHELL_HEAD + LEDE + body.strip() + FOOT
    io.open(os.path.join(OUT, "page.html"), "w", encoding="utf-8").write(page)

    css = io.open(os.path.join(HERE, "src", "css", "app.css"), encoding="utf-8").read()
    css = css.replace("html, body { height: 100%; margin: 0; }", ".appwin { height: 100%; }")
    css = css.replace("'Pretendard Variable', Pretendard, 'Malgun Gothic', 'Apple SD Gothic Neo', system-ui, sans-serif",
                      "'Pretendard Variable', Pretendard, Inter, 'Noto Sans KR', 'Malgun Gothic', system-ui, sans-serif")
    css = re.sub(r"(?m)^body \{", ".appwin {", css, count=1)
    io.open(os.path.join(OUT, "css", "app.css"), "w", encoding="utf-8").write(css)

    print("wrote", os.path.join(OUT, "page.html"), "and", os.path.join(OUT, "css", "app.css"))
    print("body rules moved to .appwin:", ".appwin { height: 100%; }" in css)


if __name__ == "__main__":
    main()
