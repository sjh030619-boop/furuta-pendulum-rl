"""앱 아이콘(build/icon.ico)을 그립니다.

메뉴바 로고와 같은 모양입니다 — 진자(공 + 막대)와 받침. 외부 이미지 파일을 두지 않고
코드로 그리는 이유는, 크기마다 선 굵기를 따로 맞춰야 작은 크기(16·24px)에서 뭉개지지
않기 때문입니다. 큰 그림을 줄이면 막대가 사라집니다.

    python app/build/make_icon.py
"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [16, 24, 32, 48, 64, 128, 256]

BG1 = (77, 151, 255)
BG2 = (51, 115, 204)
WHITE = (255, 255, 255)
BOB = (194, 65, 12)


def rounded_bg(n, ss):
    img = Image.new("RGBA", (n * ss, n * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = n * ss
    for y in range(w):
        t = y / max(1, w - 1)
        c = tuple(int(BG1[i] + (BG2[i] - BG1[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c + (255,))
    mask = Image.new("L", (w, w), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, w - 1], radius=int(w * 0.22), fill=255)
    img.putalpha(mask)
    return img


def draw(n):
    """크기 n 픽셀짜리 한 장. 작을수록 요소를 단순하게 만듭니다."""
    ss = 8 if n <= 64 else 4
    w = n * ss
    img = rounded_bg(n, ss)
    d = ImageDraw.Draw(img)

    cx = w * 0.5
    small = n <= 24

    base_w = w * 0.62
    base_h = w * 0.115
    base_y = w * 0.79
    d.rounded_rectangle([cx - base_w / 2, base_y, cx + base_w / 2, base_y + base_h],
                        radius=base_h / 2, fill=WHITE + (235,))

    rod_w = w * (0.13 if small else 0.105)
    rod_top = w * 0.33
    d.rounded_rectangle([cx - rod_w / 2, rod_top, cx + rod_w / 2, base_y + base_h * 0.2],
                        radius=rod_w / 2, fill=WHITE + (245,))

    r = w * (0.20 if small else 0.175)
    cy = w * 0.27
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE)
    if not small:
        ri = r * 0.46
        d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=BOB + (255,))

    return img.resize((n, n), Image.LANCZOS)


def main():
    imgs = [draw(n) for n in SIZES]
    out = os.path.join(HERE, "icon.ico")
    imgs[-1].save(out, format="ICO", sizes=[(n, n) for n in SIZES])
    imgs[-1].save(os.path.join(HERE, "icon.png"), format="PNG")
    print("wrote", out, "sizes", SIZES)


if __name__ == "__main__":
    main()
