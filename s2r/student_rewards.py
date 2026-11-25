"""학생이 블록으로 쌓을 만한 보상(점수 규칙) 예시.

앱(app/src/js/blockdefs.js)의 블록만으로 만들 수 있는 것만 넣었습니다. 서명도 앱과 같습니다.
    fn(phi, ald, thd, u, th) -> 점수
    phi  진자 각도 (똑바로 서면 0, 매달리면 ±3.14)   ald  진자 빠르기 [rad/s]
    thd  팔 빠르기 [rad/s]   u  모터 힘 (-1~1)   th  팔 각도 [rad]
blocks 는 앱에서 쌓는 모양 그대로 적은 글입니다(모자 블록 '점수 규칙' 아래로 한 줄씩).
kind: good = 세울 것으로 기대, hard = 될지 모름(학습지의 '아주 강하게 하면?'), bad = 실수·장난(세우면 안 됨).
"""
import math


def height(phi):
    return (1.0 + math.cos(phi)) / 2.0


def upright(phi, ald):
    return 1.0 if (abs(phi) < 0.1745 and abs(ald) < 2.0) else 0.0


REWARDS = [
    dict(key="height", name="높이만", kind="good",
         blocks=["점수 더하기 (진자 높이)"],
         intent="진자가 높이 올라갈수록 점수",
         fn=lambda phi, ald, thd, u, th: height(phi)),
    dict(key="upright_only", name="섰으면 1점", kind="good",
         blocks=["점수 더하기 (똑바로 섰나?)"],
         intent="서 있을 때만 점수 (드문 보상)",
         fn=lambda phi, ald, thd, u, th: upright(phi, ald)),
    dict(key="angle_pen", name="기울면 감점", kind="good",
         blocks=["점수 빼기 (절댓값 (진자 각도))"],
         intent="똑바로에서 벗어난 만큼 감점 (점수가 늘 0 이하)",
         fn=lambda phi, ald, thd, u, th: -abs(phi)),
    dict(key="height_bonus", name="높이 + 서면 보너스", kind="good",
         blocks=["점수 더하기 (진자 높이)", "점수 더하기 (2 × (똑바로 섰나?))"],
         intent="올라가는 것도 칭찬, 서면 크게 칭찬",
         fn=lambda phi, ald, thd, u, th: height(phi) + 2.0 * upright(phi, ald)),
    dict(key="height_effort", name="높이 - 힘 아끼기", kind="good",
         blocks=["점수 더하기 (진자 높이)", "점수 빼기 (0.1 × (제곱 (모터 힘)))"],
         intent="세우되 모터를 덜 쓰게",
         fn=lambda phi, ald, thd, u, th: height(phi) - 0.1 * u * u),
    dict(key="height_effort_strong", name="높이 - 힘 아주 아끼기", kind="hard",
         blocks=["점수 더하기 (진자 높이)", "점수 빼기 (1 × (제곱 (모터 힘)))"],
         intent="학습지 4쪽: 힘 아끼기를 아주 강하게 하면?",
         fn=lambda phi, ald, thd, u, th: height(phi) - 1.0 * u * u),
    dict(key="height_smooth", name="높이 - 진자 천천히", kind="hard",
         blocks=["점수 더하기 (진자 높이)", "점수 빼기 (0.01 × (제곱 (진자 빠르기)))"],
         intent="진자가 휙휙 돌지 않게",
         fn=lambda phi, ald, thd, u, th: height(phi) - 0.01 * ald * ald),
    dict(key="height_armhome", name="높이 - 팔 제자리", kind="good",
         blocks=["점수 더하기 (진자 높이)", "점수 빼기 (0.2 × (절댓값 (팔 각도)))"],
         intent="팔이 처음 자리에서 멀리 가지 않게",
         fn=lambda phi, ald, thd, u, th: height(phi) - 0.2 * abs(th)),
    dict(key="height_armslow", name="높이 - 팔 천천히", kind="good",
         blocks=["점수 더하기 (진자 높이)", "점수 빼기 (0.02 × (제곱 (팔 빠르기)))"],
         intent="팔이 덜 휘둘리게 (조용한 키트)",
         fn=lambda phi, ald, thd, u, th: height(phi) - 0.02 * thd * thd),
    dict(key="quiet_stand", name="조용히 서기", kind="good",
         blocks=["점수 더하기 (똑바로 섰나?)", "점수 빼기 (0.3 × (절댓값 (모터 힘)))"],
         intent="서 있을 때만 점수, 모터 힘은 늘 감점",
         fn=lambda phi, ald, thd, u, th: upright(phi, ald) - 0.3 * abs(u)),
    dict(key="greedy", name="전부 다", kind="good",
         blocks=["점수 더하기 (진자 높이)", "점수 더하기 (똑바로 섰나?)", "점수 빼기 (0.05 × (제곱 (모터 힘)))",
                 "점수 빼기 (0.002 × (제곱 (진자 빠르기)))", "점수 빼기 (0.1 × (절댓값 (팔 각도)))"],
         intent="생각나는 조건을 다 넣은 규칙",
         fn=lambda phi, ald, thd, u, th: (height(phi) + upright(phi, ald) - 0.05 * u * u
                                          - 0.002 * ald * ald - 0.1 * abs(th))),
    dict(key="mistake_sign", name="실수: 부호 반대", kind="bad",
         blocks=["점수 빼기 (진자 높이)"],
         intent="더하기 대신 빼기를 붙인 실수 (매달려 있어야 최고점)",
         fn=lambda phi, ald, thd, u, th: -height(phi)),
    dict(key="spin", name="장난: 빨리 돌려라", kind="bad",
         blocks=["점수 더하기 (절댓값 (진자 빠르기))"],
         intent="진자가 빨리 돌수록 점수 (키트 안전 정지에 걸려야 함)",
         fn=lambda phi, ald, thd, u, th: abs(ald)),
]
BY_KEY = {r["key"]: r for r in REWARDS}
