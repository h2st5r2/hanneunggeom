"""CSV와 마크다운 보고서를 만든다. 대시보드는 build_dashboard.py가 이어서 만든다.

실행: python3 -I analysis/scripts/build.py   (저장소 루트에서)
"""
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute as C  # noqa: E402

OUT = os.path.join(C.ROOT, "analysis")
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)

R = C.compute_all()
SUBS = list(C.SUBSETS)
NAMES = {tid: name for tid, name, _ in C.TABLES}
SHORT = {"제1장 선사 문화": "선사", "제2장 한국 고대사": "고대", "제3장 고려시대사": "고려",
         "제4장 조선시대사": "조선", "제5장 한국 근대사": "근대", "제6장 일제 강점기": "일제",
         "제7장 한국 현대사": "현대", "제8장 복합 문항": "복합", "특별 부록": "부록"}
N_ALL = len(C.Q)
N_CONF = len(C.SUBSETS["확인"])


# ------------------------------------------------------------------ CSV
def clean_for_csv(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            v = df[c].dropna()
            if len(v) and (v == v.round()).all():
                df[c] = df[c].round().astype("Int64")
            else:
                df[c] = df[c].round(6)
    return df


for tid, name, _ in C.TABLES:
    for s in SUBS:
        clean_for_csv(R[tid][s]).to_csv(os.path.join(TAB, f"{name}_{s}.csv"), index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ 마크다운 도우미
def fmt(v, col):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return "-"
    if isinstance(v, str):
        return v.replace("|", "\\|")
    if isinstance(v, (int,)) or (hasattr(v, "dtype") and "int" in str(v.dtype)):
        return str(int(v))
    if isinstance(v, float):
        if any(k in col for k in ("비중", "비율", "적중률")):
            return f"{v:.1f}"
        if any(k in col for k in ("평균", "회차당", "배율", "도래비", "z값", "변화", "표준편차")):
            return f"{v:.2f}"
        return f"{v:g}"
    return str(v)


def table(df, cols=None, n=None, rename=None):
    if len(df) == 0:
        return "해당 행 없음\n"
    if cols:
        df = df[cols]
    if n:
        df = df.head(n)
    rename = rename or {}
    heads = [rename.get(c, c) for c in df.columns]
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join(["---"] * len(heads)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], c) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def two(tid, cols=None, n=None, filt=None, rename=None, note=None):
    """전체/확인 표를 차례로 출력한다."""
    out = []
    for s in SUBS:
        df = R[tid][s]
        if filt:
            df = filt(df)
        label = "전체 1,650문항" if s == "전체" else f"status가 확인인 {N_CONF}문항"
        out.append(f"**{label}**\n\n" + table(df, cols, n, rename))
    return "\n".join(out)


def merged(tid, keys, vals, rename=None):
    a, c = R[tid]["전체"], R[tid]["확인"]
    m = a[keys + vals].merge(c[keys + vals], on=keys, how="left", suffixes=("(전체)", "(확인)"))
    return m


def mtable(m, keys, vals, rename=None):
    cols = keys + [f"{v}({s})" for v in vals for s in SUBS]
    return table(m, cols, rename=rename)


def j(word, with_final, without_final):
    code = ord(word[-1]) - 0xAC00
    return with_final if 0 <= code < 11172 and code % 28 != 0 else without_final


def j2(word):
    code = ord(word[-1]) - 0xAC00
    return "로" if not (0 <= code < 11172) or code % 28 in (0, 8) else "으로"


def shortcols(df):
    return df.rename(columns=SHORT)


# ------------------------------------------------------------------ 공통 사실
def ch_share_conf():
    d = R["L1b"]["확인"].set_index("시대")["문항수"]
    early = int(d[C.CHAPTERS[:4]].sum())
    late = int(d[C.CHAPTERS[4:7]].sum())
    return early, late, int(d.sum())


EARLY, LATE, _ = ch_share_conf()
from collections import Counter as _Ctr
ROLE = _Ctr(c["role"] for x in C.Q for c in x["ch"])
N_ARR_Q = sum(1 for x in C.Q if x["ch"][0]["role"] == "배열")
N_NEG_Q = sum(1 for x in C.Q if x["neg"])
N_ANS_Q = {k: sum(1 for x in v if C.ans_choice(x)) for k, v in C.SUBSETS.items()}
NO_ABC = [c for c in C.SUB_NAME if c not in C.TAX["abc"]]

SCOPE_NOTE = f"""## 집계 기준

- 자료: `questions.json` 47~79회 {len(C.ROUNDS)}회차 × 50문항 = {N_ALL:,}문항, `taxonomy.json`.
- "전체"는 {N_ALL:,}문항, "확인"은 status가 "확인"인 {N_CONF}문항이다. 모든 표를 두 기준으로 계산했고, 전체 표는 `analysis/tables/`의 CSV에 있다.
- **확인 문항의 시대 편중**: 확인 {N_CONF}문항 중 제1~4장(선사~조선)이 {EARLY}문항({100*EARLY/N_CONF:.1f}%)이고 제5~7장(근대·일제·현대)은 {LATE}문항({100*LATE/N_CONF:.1f}%)이다. 확인 기준의 시대 비중과 소주제 순위는 검수가 끝난 범위의 값이며, 시험 전체의 경향으로 읽으면 근대 이후가 과소 집계된다. 문항 번호대 표에서는 31번 이후가 같은 이유로 비어 있다.
- "시대"는 `taxonomy.json`의 장(제1~7장, 제8장 복합 문항, 특별 부록)이고, "대주제"는 `topic`(51개), "소주제"는 `code`(174개)이다.
- 선지 role은 요청에 따라 "정답"과 "오답"만 쓴다. 그 밖의 role은 집계에서 뺐다: 배열 선지 {ROLE['배열']}개(순서·연표 등 {N_ARR_Q}문항), 부정형 문항의 "정답(부정형 옳은 선지)" {ROLE['정답(부정형 옳은 선지)']}개와 "오답(부정형 정답)" {ROLE['오답(부정형 정답)']}개(부정형 {N_NEG_Q}문항). 부정형 문항의 답은 틀린 진술이므로 단서→정답 짝과 함정 선지에서 빠진다. 정답 role이 있는 문항은 전체 {N_ANS_Q['전체']:,}문항, 확인 {N_ANS_Q['확인']}문항이다.
- 키워드(`kw`, `k`)는 쉼표로 나눠 한 단어씩 센다(괄호 안 쉼표는 나누지 않음). 표기가 다른 같은 개념은 합치지 않았다.
- 소수는 반올림해 표시하고, 합계가 어긋나 보이면 CSV의 원값이 기준이다.
"""

REPORTS = {}

# ================================================================== 학습자용
L = R
l1b, l1a = L["L1b"]["전체"], L["L1a"]["전체"]
top_ch = l1b.sort_values(["문항수", "시대"], ascending=[False, True]).head(3)
l2 = L["L2"]["전체"]
l2c = L["L2"]["확인"]
zero_all = l2[l2["출제횟수"] == 0]
l3 = L["L3"]["전체"].sort_values(["문항수", "소주제", "자료키워드", "정답키워드"], ascending=[False, True, True, True])
l4a = L["L4a"]["전체"]
l4b = L["L4b"]["전체"]
l5a = L["L5a"]["전체"]
l5b = L["L5b"]["전체"]
l6 = L["L6"]["전체"]
l6b = L["L6b"]["전체"].iloc[-1]
l6bc = L["L6b"]["확인"].iloc[-1]
l6_cand = {s: int((L["L6"][s]["후보"] == "예").sum()) for s in SUBS}
flag_a = {s: int((L["L5a"][s]["판정"] == "집중").sum()) for s in SUBS}
flag_b = {s: int((L["L5b"][s]["판정"] == "집중").sum()) for s in SUBS}
top1 = l2.iloc[0]
cnt_nonzero = {s: int((R["L2"][s]["출제횟수"] > 0).sum()) for s in SUBS}

learner_summary = [
    f"시대별로는 {top_ch.iloc[0]['시대']}{j(top_ch.iloc[0]['시대'], '이', '가')} 회차당 {top_ch.iloc[0]['회차당평균']:.1f}문항({top_ch.iloc[0]['비중']:.1f}%)으로 가장 많고, "
    f"{top_ch.iloc[1]['시대']} {top_ch.iloc[1]['회차당평균']:.1f}문항, {top_ch.iloc[2]['시대']} {top_ch.iloc[2]['회차당평균']:.1f}문항 순이다(50문항 기준, 전체).",
    f"소주제 1위는 '{top1['소주제명']}'({top1['소주제']})로 {int(top1['출제횟수'])}문항이 {int(top1['출제회차수'])}개 회차에 나왔고 최근 출제는 {int(top1['최근출제회차'])}회이다. "
    f"174개 소주제 중 {cnt_nonzero['전체']}개가 출제됐고 {len(zero_all)}개('{zero_all.iloc[0]['소주제명']}', '{zero_all.iloc[1]['소주제명']}')는 한 번도 나오지 않았다.",
    f"가장 자주 짝지어진 단서→정답은 '{l3.iloc[0]['자료키워드']}'→'{l3.iloc[0]['정답키워드']}'({l3.iloc[0]['소주제명']}, {int(l3.iloc[0]['문항수'])}문항)이다. "
    f"오답 선지에 가장 많이 쓰인 키워드는 '{l4a.iloc[0]['키워드']}'(오답 {int(l4a.iloc[0]['오답문항수'])}문항, 정답 {int(l4a.iloc[0]['정답문항수'])}문항)이고, 정답과 가장 자주 함께 오답이 되는 짝은 '{l4b.iloc[0]['정답소주제명']}'{j(l4b.iloc[0]['정답소주제명'], '과', '와')} '{l4b.iloc[0]['오답소주제명']}'({int(l4b.iloc[0]['문항수'])}문항)이다.",
    f"최근 10회(70~79회)에는 소주제 {flag_a['전체']}개(최근 4문항 이상)와 키워드 {flag_b['전체']}개(최근 3문항 이상)가 이전 23회보다 회차당 1.5배 이상 자주 나왔다(이전 0회 포함). 소주제 중 최근 출제 수가 가장 많은 것은 '{l5a.iloc[0]['소주제명']}'({int(l5a.iloc[0]['최근10회문항수'])}문항, 배율 {l5a.iloc[0]['배율']:.2f})이다.",
    f"재출제 간격 규칙(출제 4회 이상, 마지막 출제 후 경과 회차 ≥ 간격 중앙값)으로 80회 후보 키워드 {l6_cand['전체']}개를 뽑았다. 70~79회 백테스트 적중률은 {l6b['후보적중률']:.1f}%로 "
    f"같은 개수의 빈도 상위 키워드({l6b['빈도상위적중률']:.1f}%)와 전체 대상 평균({l6b['전체대상적중률']:.1f}%)보다 낮아, 후보 목록은 참고용이다.",
]

s = []
s.append("# 한국사능력검정시험 심화 출제 경향 분석: 학습자용\n")
s.append("## 핵심 요약\n")
s.append("\n".join(f"{i}. {t}" for i, t in enumerate(learner_summary, 1)) + "\n")
s.append(f"> 확인 문항 {N_CONF}개는 선사~조선 시대에 {100*EARLY/N_CONF:.1f}%가 몰려 있다. 아래 \"확인\" 표는 근대 이후 소주제를 거의 포함하지 않는다.\n")
s.append(SCOPE_NOTE)

# 1
s.append("## 1. 시대별 출제 비중\n")
s.append("회차당 평균은 해당 시대(대주제) 문항 수를 33회차로 나눈 값이다. 전체 기준은 한 회차 50문항 중 몇 문항인지를 뜻한다. 확인 기준은 확인 문항만 센 값이라 합이 50보다 작다.\n")
V = ["문항수", "회차당평균", "비중"]
m = merged("L1b", ["시대"], V)
s.append("**시대(장)별**\n\n" + mtable(m, ["시대"], V))
m = merged("L1a", ["대주제", "대주제명"], V)
s.append("**대주제(topic)별**\n\n" + mtable(m, ["대주제", "대주제명"], V))

# 2
s.append("## 2. 소주제별 빈출 순위\n")
s.append("출제횟수는 `code`가 해당 소주제인 문항 수이고, 미출제 회차 수는 79회에서 최근 출제 회차를 뺀 값이다. 동률은 최근 출제 회차, 소주제 코드 순으로 정했다. 전체 174개 소주제의 순위는 CSV에 있다.\n")
s.append(two("L2", ["순위", "소주제", "소주제명", "출제횟수", "출제회차수", "최근출제회차", "미출제회차수"], n=20))
s.append(f"출제 0회 소주제: 전체 기준 {int((R['L2']['전체']['출제횟수']==0).sum())}개, 확인 기준 {int((R['L2']['확인']['출제횟수']==0).sum())}개(확인 기준은 위 편중 때문에 근대 이후 소주제가 대부분 0회).\n")

# 3
s.append("## 3. 단서 → 정답 짝\n")
s.append("한 문항의 자료 키워드(`kw`) 하나와 정답 선지 키워드(`k`) 하나를 짝으로 보고, 문항 수를 소주제(문항의 `code`)별로 센다. 정답 role이 있는 문항만 대상이다. 아래는 정답 role 문항이 많은 소주제 12개의 상위 3개 짝이고, " + f"{len(R['L3']['전체']):,}쌍(전체)·{len(R['L3']['확인']):,}쌍(확인)의 전체 목록은 CSV에 있다.\n")


def l3_filter(df):
    top = (df.drop_duplicates("소주제").sort_values(["소주제정답문항수", "소주제"], ascending=[False, True]).head(12)["소주제"])
    sub = df[df["소주제"].isin(top) & (df["소주제내순위"] <= 3)].copy()
    sub["_o"] = sub["소주제"].map({c: i for i, c in enumerate(top)})
    return sub.sort_values(["_o", "소주제내순위"]).drop(columns="_o")


s.append(two("L3", ["소주제", "소주제명", "소주제정답문항수", "소주제내순위", "자료키워드", "정답키워드", "문항수"], filt=l3_filter))

# 4
s.append("## 4. 함정 선지\n")
s.append("### 오답 선지에 자주 나온 키워드\n")
s.append("오답 role 선지의 키워드를 문항 단위로 센다. 같은 키워드가 정답 선지로도 나온 문항 수와 비교해 오답 비율을 구했다. 주 소주제는 그 키워드가 붙은 오답 선지의 소주제 중 가장 많은 것이다.\n")
s.append(two("L4a", ["순위", "키워드", "오답문항수", "정답문항수", "오답비율", "주소주제", "소주제명"], n=20))
s.append("### 정답 소주제와 함께 오답으로 묶이는 소주제 짝\n")
sp = {s_: C.self_pair_count(C.SUBSETS[s_]) for s_ in SUBS}
s.append(f"정답 선지의 소주제(`ch.code`)와 같은 문항의 오답 선지 소주제를 짝지어 문항 수를 센다. 한 문항에서 같은 짝은 한 번만 센다. 오답 선지가 정답과 같은 소주제인 문항(전체 {sp['전체']}문항, 확인 {sp['확인']}문항)은 짝에서 뺐다. 정답소주제문항수는 그 소주제가 정답 선지로 나온 문항 수이다.\n")
s.append(two("L4b", ["순위", "정답소주제", "정답소주제명", "오답소주제", "오답소주제명", "문항수", "정답소주제문항수", "같은대주제"], n=15))

# 5
s.append("## 5. 최근 10회(70~79회) 집중 출제\n")
s.append("회차당 문항 수를 최근 10회와 이전 23회(47~69회)로 나눠 배율(최근 ÷ 이전)을 구했다. 소주제는 최근 4문항 이상, 키워드는 최근 3문항 이상이면서 배율 1.5 이상(이전 0회 포함)일 때 \"집중\"으로 표시했다. 배율의 \"-\"는 이전 23회에 한 번도 나오지 않은 경우이다.\n")
s.append("### 소주제\n")
s.append(two("L5a", ["소주제", "소주제명", "최근10회문항수", "이전23회문항수", "최근회차당", "이전회차당", "배율"], filt=lambda d: d[d["판정"] == "집중"]))
s.append("### 키워드\n")
s.append(two("L5b", ["키워드", "최근10회문항수", "이전23회문항수", "최근회차당", "이전회차당", "배율"], filt=lambda d: d[d["판정"] == "집중"], n=20))

# 6
s.append("## 6. 재출제 간격으로 본 80회 후보 키워드\n")
s.append("**규칙**: 자료 키워드가 서로 다른 4개 회차 이상에 나왔고, 79회까지의 경과 회차(80 − 최근 출제 회차)가 출제 간격 중앙값 이상(도래비 ≥ 1)이면 후보이다. 후보 안에서는 출제 회차 수, 도래비 순으로 정렬했다.\n")
s.append(two("L6", ["순위", "키워드", "출제회차수", "최근출제회차", "간격중앙값", "경과회차", "도래비", "최근10회출제회차수"],
             filt=lambda d: d[d["후보"] == "예"], n=25))
s.append("**규칙 검증(백테스트)**: 같은 규칙을 70~79회 각 회차에 적용해, 그 회차 직전까지의 자료만으로 뽑은 후보가 실제로 그 회차의 자료 키워드로 나왔는지 확인했다. 비교 기준은 (a) 같은 대상 중 출제 회차 수가 많은 순으로 후보와 같은 개수를 뽑은 것, (b) 대상 키워드 전체이다.\n")
s.append(two("L6b"))
s.append(f"합계 기준 후보 적중률은 전체 {l6b['후보적중률']:.1f}%, 확인 {l6bc['후보적중률']:.1f}%이고, 빈도 상위 동수 기준은 {l6b['빈도상위적중률']:.1f}%, {l6bc['빈도상위적중률']:.1f}%이다. 간격 규칙은 단순 빈도보다 낫다는 근거를 얻지 못했으므로, 후보를 고를 때는 출제 회차 수가 많은 키워드를 우선하는 편이 이 데이터에서는 더 맞았다. 표본이 10회차뿐이라 차이의 크기는 불확실하다.\n")

REPORTS["학습자용.md"] = "\n".join(s)

# ================================================================== 교수자용
p1c = R["P1c"]["전체"]
p1b = R["P1b"]["전체"]
t4 = p1c[p1c["시대"] == "제4장 조선시대사"].iloc[0]
mx = p1b.sort_values(["최대", "대주제"], ascending=[False, True]).iloc[0]
p2d = R["P2d"]
inv = {s_: p2d[s_]["역행횟수"].mean() for s_ in SUBS}
p2a = R["P2a"]["전체"]
p3a, p3aC = R["P3a"]["전체"], R["P3a"]["확인"]
p3a_zero = int((p3a["문항수"] == 0).sum())
p3a_zeroC = int((p3aC["문항수"] == 0).sum())
p3d = R["P3d"]["전체"]
stale = p3d[p3d["미출제회차수"].notna()].iloc[0]
p4a = R["P4a"]["전체"]
p4c = R["P4c"]["전체"]
neg_total = int(p4c["부정형"].sum())
set_total = int(p4c["세트문항"].sum())
neg_hi = int(p4c[p4c["번호대"].isin(["21~30번", "41~50번"])]["부정형"].sum())
p5a = R["P5a"]["전체"]
inc = p5a.sort_values(["변화_C_A", "소주제"], ascending=[False, True]).iloc[0]
dec = p5a.sort_values(["변화_C_A", "소주제"], ascending=[True, True]).iloc[0]
p5b = R["P5b"]["전체"]
p6b = R["P6b"]["전체"].iloc[0]
p6b_c = R["P6b"]["확인"].iloc[0]
p6 = R["P6"]["전체"]
nz = {s_: int((R["P5a"][s_]["z값"].abs() >= 2).sum()) for s_ in SUBS}

# 제4장(조선) 중앙번호 구간 변화를 요약에 사용
p2c = R["P2c"]["전체"]
ch_med = p2c[p2c["시대"] == "제4장 조선시대사"].set_index("구간")["중앙번호평균"]

prof_summary = [
    f"{C.CHAPTERS[3]}{j(C.CHAPTERS[3], '이', '가')} 회차당 평균 {t4['평균']:.1f}문항(최소 {int(t4['최소'])}, 최대 {int(t4['최대'])})으로 가장 많고, 대주제 중 한 회차 최대치는 '{mx['대주제명']}'({mx['대주제']})의 {int(mx['최대'])}문항이다(전체 33회차).",
    f"번호대별 최다 시대는 " + ", ".join(f"{r['번호대']} {SHORT[max(C.CHAPTERS, key=lambda c: (r[c], -C.CH_RANK[c]))]}({100*max(r[c] for c in C.CHAPTERS)/r['합계']:.1f}%)" for _, r in p2a.iterrows()) + f"이다. 시대 순서는 한 회차에 평균 {inv['전체']:.1f}번 거꾸로 뛰고(확인 기준 {inv['확인']:.1f}번), 조선시대사의 중앙 번호는 47~59회 {ch_med['47~59회']:.1f}번, 60~69회 {ch_med['60~69회']:.1f}번, 70~79회 {ch_med['70~79회']:.1f}번이다.",
    f"세분류 묶음 {len(p3a)}개 중 {p3a_zero}개({100*p3a_zero/len(p3a):.1f}%)는 한 번도 출제되지 않았고(확인 기준 {p3a_zeroC}개), 소주제 중 미출제 기간이 가장 긴 출제 소주제는 '{stale['소주제명']}'(마지막 출제 {int(stale['최근출제회차'])}회, 이후 {int(stale['미출제회차수'])}회 동안 미출제)이다.",
    f"발문 유형은 '{p4a.iloc[0]['유형']}'{j(p4a.iloc[0]['유형'], '이', '가')} {int(p4a.iloc[0]['문항수']):,}문항({p4a.iloc[0]['비중']:.1f}%)이고, 부정형 {neg_total}문항 중 {neg_hi}문항이 21~30번과 41~50번에 있다. 세트 문항 {set_total}문항 중 {int(p4c.iloc[4]['세트문항'])}문항이 41~50번이다.",
    f"47~59회 대비 70~79회에 회차당 문항이 가장 늘어난 소주제는 '{inc['소주제명']}'({inc['변화_C_A']:+.2f}), 가장 줄어든 소주제는 '{dec['소주제명']}'({dec['변화_C_A']:+.2f})이다. "
    f"같은 선지 문장이 둘 이상의 회차에 다시 나온 사례는 {int(p6b['둘이상회차에등장한문장수']):,}문장이며, 이 문장들이 전체 선지(배열 선지 제외)의 {p6b['선지중비율']:.1f}%를 차지한다.",
]

s = []
s.append("# 한국사능력검정시험 심화 출제 경향 분석: 교수자용\n")
s.append("## 핵심 요약\n")
s.append("\n".join(f"{i}. {t}" for i, t in enumerate(prof_summary, 1)) + "\n")
s.append(f"> 확인 문항 {N_CONF}개는 선사~조선 시대에 {100*EARLY/N_CONF:.1f}%가 몰려 있어, 확인 기준의 번호대·시대 표는 31번 이후 문항이 거의 없다.\n")
s.append(SCOPE_NOTE)

# 1
s.append("## 1. 회차별 시대 배분표\n")
s.append("33회차 각각에서 시대(대주제) 문항 수를 센 뒤 평균·최소·최대를 구했다(문항이 없는 회차는 0으로 포함). 회차 × 대주제 원자료(33행 × 51열)는 `교수1_회차별_대주제문항수_*.csv`에 있다.\n")
V = ["평균", "최소", "최대"]
m = merged("P1c", ["시대"], V)
s.append("**시대(장)별**\n\n" + mtable(m, ["시대"], V))
m = merged("P1b", ["대주제", "대주제명"], V)
s.append("**대주제별**\n\n" + mtable(m, ["대주제", "대주제명"], V))

# 2
s.append("## 2. 문항 번호대와 시대의 대응, 회차별 경계 변화\n")
s.append("### 번호대 × 시대 문항 수\n")
s.append(two("P2a", rename=SHORT))
s.append("### 시대 순서의 역행\n")
s.append("제1~7장 문항을 번호순으로 놓았을 때 바로 앞 문항보다 시대가 앞서는 횟수(역행)를 회차별로 센 값이다. 제8장(복합)·부록은 제외했다. 구간별 평균은 다음과 같다.\n")
rows = []
for name, lo, hi in C.PERIODS:
    row = {"구간": name}
    for s_ in SUBS:
        d = R["P2d"][s_]
        d = d[(d["회차"] >= lo) & (d["회차"] <= hi)]["역행횟수"]
        row[f"평균({s_})"] = d.mean()
        row[f"최대({s_})"] = int(d.max())
    rows.append(row)
s.append(table(pd.DataFrame(rows)))
s.append("### 구간별 시대 경계\n")
s.append("구간 안 회차들에서 각 시대 문항의 첫 번호·마지막 번호·중앙 번호를 구해 평균했다. 회차별 값은 `교수2_회차별_시대경계_*.csv`에 있다. 확인 기준은 확인 문항만으로 구한 번호라 31번 이후 시대는 값이 없거나 소수 문항에 의존한다.\n")
s.append(two("P2c", ["구간", "시대", "등장회차수", "회차당문항수", "첫번호평균", "마지막번호평균", "중앙번호평균"]))

# 3
s.append("## 3. 세분류(abc) 묶음 출제와 미출제\n")
s.append("`taxonomy.json`의 `abc`에는 소주제 " + f"{len(C.TAX['abc'])}개에 걸쳐 묶음 {len(p3a)}개가 있고, 문항의 `blk`가 묶음을 가리킨다. `abc`가 없는 소주제는 {len(NO_ABC)}개이다({', '.join(NO_ABC)}).\n")
s.append("### 배정 현황\n")
s.append(two("P3c"))
s.append("### 묶음별 출제 수 요약\n")
rows = []
for s_ in SUBS:
    d = R["P3a"][s_]
    rows.append({"기준": s_, "묶음수": len(d), "출제된 묶음": int((d["문항수"] > 0).sum()), "미출제 묶음": int((d["문항수"] == 0).sum()),
                 "묶음당 평균 문항": d["문항수"].mean(), "최대 문항": int(d["문항수"].max())})
s.append(table(pd.DataFrame(rows)))
s.append("묶음별 출제 수 전체 목록은 `교수3_세분류묶음별_출제수_*.csv`에 있다. 문항 수가 많은 묶음 상위 10개는 다음과 같다.\n")
s.append(two("P3a", ["소주제", "소주제명", "묶음", "묶음명", "문항수", "출제회차수", "최근출제회차"],
             filt=lambda d: d.sort_values(["문항수", "소주제", "묶음"], ascending=[False, True, True]), n=10))
s.append("### 한 번도 안 나온 묶음\n")
s.append("소주제별로 미출제 묶음을 모았다. 마지막 열은 같은 소주제에서 다른 묶음이 출제된 문항 수이다.\n")


def zero_grouped(df):
    rows = []
    for code, g in df.groupby("소주제", sort=False):
        rows.append({"소주제": code, "소주제명": g.iloc[0]["소주제명"],
                     "미출제 묶음": "; ".join(f"{r['묶음']} {r['묶음명']}" for _, r in g.iterrows()),
                     "미출제 묶음 수": len(g), "소주제내 다른 묶음 문항수": int(g.iloc[0]["소주제내_다른묶음문항수"])})
    return pd.DataFrame(rows)


for s_ in SUBS:
    label = "전체 1,650문항" if s_ == "전체" else f"status가 확인인 {N_CONF}문항"
    s.append(f"**{label}**\n\n" + table(zero_grouped(R["P3b"][s_])))
s.append("### 오래 안 나온 소주제\n")
s.append("미출제 회차 수는 79회에서 마지막 출제 회차를 뺀 값이다. 한 번도 출제되지 않은 소주제는 맨 위에 두었다.\n")
s.append(two("P3d", ["순위", "소주제", "소주제명", "출제횟수", "최근출제회차", "미출제회차수"], n=20))

# 4
s.append("## 4. 발문 유형 분포와 부정형·세트 문항의 위치\n")
s.append("### 유형 분포\n")
s.append(two("P4a", ["유형", "문항수", "비중", "회차당평균", "부정형문항수", "주요시대", "주요시대문항수", "주요시대내비중", "주요대주제", "주요대주제문항수"]))
s.append("### 유형 × 시대\n")
s.append(two("P4b", rename=SHORT))
s.append("### 번호대별 부정형·세트\n")
s.append(two("P4c"))
negn = R["P4d"]["전체"].sort_values(["부정형", "번호"], ascending=[False, True]).head(8)
s.append("부정형이 많이 나온 번호(전체, 33회차 합): " + ", ".join(f"{int(r['번호'])}번 {int(r['부정형'])}문항" for _, r in negn.iterrows()) + ". 번호별 전체 목록은 `교수4_번호별_부정형세트_*.csv`에 있다.\n")
s.append("### 세트 문항\n")
s.append("`set` 값은 함께 묶인 문항 번호이다(예: 47-48은 47·48번이 한 세트).\n")
s.append(two("P4e"))

# 5
s.append("## 5. 구간별(47~59회, 60~69회, 70~79회) 증감\n")
s.append("회차당 문항 수를 구간별로 구하고(구간 문항 수 ÷ 구간 회차 수: 13·10·10), 70~79회와 47~59회의 차이(C−A), 60~69회와의 차이를 표시했다. z값은 두 구간의 회차당 평균 차이를 푸아송 분산으로 나눈 대략의 지표이다(|z| ≥ 2이면 우연으로 보기 어렵지만, 174개를 한꺼번에 보면 우연히도 몇 개는 그 값을 넘는다). "
         f"|z| ≥ 2인 소주제는 전체 {nz['전체']}개, 확인 {nz['확인']}개이다.\n")
PV = ["47~59회_회차당", "60~69회_회차당", "70~79회_회차당", "변화_C_A", "z값"]
s.append("### 시대(장)\n")
s.append(two("P5c", ["시대"] + PV))
s.append("### 발문 유형\n")
s.append(two("P5b", ["유형"] + PV))
s.append("### 소주제: 70~79회에 늘어난 상위 15개\n")
s.append(two("P5a", ["소주제", "소주제명"] + PV,
             filt=lambda d: d.sort_values(["변화_C_A", "소주제"], ascending=[False, True]), n=15))
s.append("### 소주제: 70~79회에 줄어든 상위 15개\n")
s.append(two("P5a", ["소주제", "소주제명"] + PV,
             filt=lambda d: d.sort_values(["변화_C_A", "소주제"], ascending=[True, True]), n=15))
s.append("174개 소주제 전체는 `교수5_구간별_소주제_*.csv`에 있다.\n")

# 6
s.append("## 6. 다른 회차에 다시 나온 선지 문장\n")
s.append("선지 문장(`t`)을 공백만 정리해 완전히 같은 것끼리 묶고, 서로 다른 회차 둘 이상에 나온 문장을 모았다. 배열 선지(\"(가) - (나) - (다)\")는 뺐다. 역할혼재는 한 문장이 옳은 진술(정답, 정답(부정형 옳은 선지))과 틀린 진술(오답, 오답(부정형 정답))로 모두 쓰인 경우이다.\n")
s.append(two("P6b"))
s.append("**여러 회차에 나온 횟수 상위 25개**\n")
s.append(two("P6", ["순위", "선지문장", "등장횟수", "등장회차수", "회차", "정답", "오답", "부정형_옳은선지", "부정형_정답", "역할혼재"], n=25))
mix = {s_: int((R["P6"][s_]["역할혼재"] == "예").sum()) for s_ in SUBS}
s.append(f"역할혼재 문장은 전체 {mix['전체']}개, 확인 {mix['확인']}개이다. 전체 목록은 `교수6_재출제_선지문장_*.csv`에 있다.\n")

REPORTS["교수자용.md"] = "\n".join(s)


# ================================================================== 시대별
E1, E3, E4, E5, E6 = (R[t]["전체"] for t in ("E1", "E3", "E4", "E5", "E6"))
chn = E5.set_index("시대")["문항수"]
big = [c for c in C.CHAPTERS if chn[c] >= 50]
tops = E1[(E1["시대내순위"] == 1) & E1["시대"].isin(big)]
t1 = tops.sort_values(["시대내비중", "시대"], ascending=[False, True]).iloc[0]
lo_, hi_ = tops["시대내비중"].min(), tops["시대내비중"].max()
short_n = {s_: int((R["E1"][s_]["표본"] == "표본 부족").sum()) for s_ in SUBS}
e4 = E4.merge(E4.groupby("소주제")["문항수"].sum().rename("소주제문항수"), left_on="소주제", right_index=True)
nb_ = E4.groupby("소주제")["묶음"].transform("count")
e4 = e4[(e4["소주제문항수"] >= 10) & (nb_.loc[e4.index] >= 3)].sort_values(["소주제내비중", "소주제"], ascending=[False, True]).iloc[0]
reg = E1[E1["출제회차수"] >= 8]
fast = reg.sort_values(["간격중앙값", "문항수", "소주제"], ascending=[True, False, True]).iloc[0]
old = E1[E1["문항수"] >= 8].sort_values(["미출제회차수", "소주제"], ascending=[False, True]).iloc[0]
e6u = E6.sort_values(["변화_C_A", "소주제"], ascending=[False, True]).iloc[0]
e6d = E6.sort_values(["변화_C_A", "소주제"], ascending=[True, True]).iloc[0]
era_summary = [
    f"시대 안에서 비중이 가장 큰 소주제는 {t1['시대']}의 '{t1['소주제명']}'{j2(t1['소주제명'])} 시대 문항의 {t1['시대내비중']:.1f}%를 차지한다. 문항 50개 이상인 {len(big)}개 시대에서 1위 소주제의 비중은 {lo_:.1f}~{hi_:.1f}%이다.",
    f"출제 문항이 5개 미만인 소주제는 전체 기준 {short_n['전체']}개, 확인 기준 {short_n['확인']}개이다. 이 소주제들의 비중과 주기는 표에서 \"표본 부족\"으로 표시했다.",
    f"문항 10개 이상이고 묶음이 3개 이상인 소주제 중 한 세분류 묶음에 가장 쏠린 곳은 '{e4['소주제명']}'의 '{e4['묶음']} {e4['묶음명']}'{j2(e4['묶음명'])}, {int(e4['소주제문항수'])}문항 중 {int(e4['문항수'])}문항({e4['소주제내비중']:.1f}%)이다.",
    f"출제 회차 8개 이상인 소주제 {len(reg)}개 중 간격 중앙값이 가장 짧은 곳은 '{fast['소주제명']}'({fast['시대']})의 {fast['간격중앙값']:g}회이다. 문항 8개 이상인 소주제 중 가장 오래 나오지 않은 곳은 '{old['소주제명']}'{j2(old['소주제명'])} 마지막 출제가 {int(old['최근출제회차'])}회이다.",
    f"47~59회 대비 70~79회에 회차당 문항이 가장 늘어난 소주제는 {e6u['시대']}의 '{e6u['소주제명']}'({e6u['변화_C_A']:+.2f}), 가장 줄어든 소주제는 {e6d['시대']}의 '{e6d['소주제명']}'({e6d['변화_C_A']:+.2f})이다.",
]
s = []
s.append("# 한국사능력검정시험 심화 출제 경향 분석: 시대별·소주제별\n")
s.append("## 핵심 요약\n")
s.append("\n".join(f"{i}. {t}" for i, t in enumerate(era_summary, 1)) + "\n")
s.append(f"> 확인 문항 {N_CONF}개는 선사~조선 시대에 {100*EARLY/N_CONF:.1f}%가 몰려 있어, 제5장 이후의 \"확인\" 표는 대부분 비어 있다.\n")
s.append(SCOPE_NOTE)
s.append("## 읽는 법\n")
s.append("- 시대마다 소주제 표, 시대 요약(키워드 상위), 세분류 묶음 상위, 구간별 증감을 차례로 둔다. 각 표는 전체와 확인 기준으로 나눠 실었다.\n"
         "- 시대내비중은 그 시대 문항 중 소주제가 차지하는 비율이다. 간격중앙값은 소주제가 출제된 서로 다른 회차 사이 간격의 중앙값(회차)이고, 미출제회차수는 79회에서 마지막 출제 회차를 뺀 값이다.\n"
         "- 표본 부족은 문항이 5개 미만이라 비중·주기를 믿기 어렵다는 표시이다.\n"
         "- 소주제마다 자료·정답·함정 키워드 상위 5개(`시대2_소주제카드_키워드_*.csv`)와 함께 오답으로 묶이는 소주제 상위 3개(`시대3_소주제카드_함정짝_*.csv`)는 CSV와 대시보드의 \"시대별\" 탭에서 볼 수 있다.\n"
         "- 괄호 안 숫자는 그 키워드가 나온 문항 수이다.\n")
EC = ["시대내순위", "소주제", "소주제명", "문항수", "시대내비중", "출제회차수", "최근출제회차", "간격중앙값", "미출제회차수", "주요유형", "표본"]
for i, ch in enumerate(C.CHAPTERS, 1):
    s.append(f"## {i}. {ch}\n")
    nall, nconf = int(R["E5"]["전체"].set_index("시대").loc[ch, "문항수"]), int(R["E5"]["확인"].set_index("시대").loc[ch, "문항수"])
    kall = int(R["E5"]["전체"].set_index("시대").loc[ch, "출제소주제수"])
    sall = int(R["E5"]["전체"].set_index("시대").loc[ch, "분류표소주제수"])
    s.append(f"전체 {nall}문항(소주제 {sall}개 중 {kall}개 출제), 확인 {nconf}문항.\n")
    s.append("### 소주제별 비중과 주기\n")
    s.append(two("E1", EC, filt=lambda d, ch=ch: d[(d["시대"] == ch) & (d["문항수"] > 0)]))
    s.append("### 시대 요약 키워드\n")
    s.append(two("E5", ["문항수", "소주제TOP5", "자료키워드TOP5", "정답키워드TOP5", "함정키워드TOP5", "주요유형TOP3"], filt=lambda d, ch=ch: d[d["시대"] == ch]))
    s.append("### 세분류 묶음 상위 8개\n")
    s.append(two("E4", ["소주제", "소주제명", "묶음", "묶음명", "문항수", "소주제내비중", "출제회차수", "최근출제회차"],
                 filt=lambda d, ch=ch: d[(d["시대"] == ch) & (d["문항수"] > 0)].sort_values(["문항수", "소주제", "묶음"], ascending=[False, True, True]), n=8))
    s.append("### 70~79회에 늘어난 소주제와 줄어든 소주제\n")
    s.append("상위·하위 각 3개이며, 회차당 문항 수 기준이다(47~59회 13회, 60~69회 10회, 70~79회 10회).\n")

    def chg(d, ch=ch):
        d = d[(d["시대"] == ch) & (d["총문항수"] > 0)]
        return pd.concat([d.head(3), d.tail(3)]).drop_duplicates("소주제")

    s.append(two("E6", ["소주제", "소주제명", "47~59회_회차당", "60~69회_회차당", "70~79회_회차당", "변화_C_A", "z값"], filt=chg))
REPORTS["시대별.md"] = "\n".join(s)

for name, text in REPORTS.items():
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)

print("CSV", len(os.listdir(TAB)), "files; reports:", {k: len(v) for k, v in REPORTS.items()})

import build_dashboard  # noqa: E402
build_dashboard.main()
