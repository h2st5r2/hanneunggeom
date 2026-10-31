"""검증: 원본 JSON에서 pandas로 독립 재계산해 CSV·마크다운·대시보드와 대조한다.

compute.py의 함수를 쓰지 않는다(분류표·키워드 분리·집계 모두 별도 구현).
실행: python3 -I analysis/scripts/verify.py   (저장소 루트에서)
"""
import glob
import json
import math
import os
import re
import statistics
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN, TAB = os.path.join(ROOT, "analysis"), os.path.join(ROOT, "analysis", "tables")
Q = json.load(open(os.path.join(ROOT, "questions.json"), encoding="utf-8"))
TAX = json.load(open(os.path.join(ROOT, "taxonomy.json"), encoding="utf-8"))

fails, passes = [], 0
log = []


def ok(cond, msg):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(msg)
        log.append("FAIL " + msg)


def toks(s):
    parts = re.split(r",\s*(?![^()]*\))", s or "")
    return list(dict.fromkeys(re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()))


chap = {t["key"]: t["chapter"] for t in TAX["topics"]}
chapters = list(dict.fromkeys(chap.values()))
subname = {s["code"]: s["name"] for t in TAX["topics"] for s in t["subs"]}
df = pd.DataFrame([{k: x[k] for k in ("id", "r", "n", "topic", "code", "status", "type", "neg", "set", "kw", "q")} | {"blk": x.get("blk") or ""} for x in Q])
df["chapter"] = df.topic.map(chap)
chs = {x["id"]: x["ch"] for x in Q}
SUB = {"전체": df, "확인": df[df.status == "확인"]}
ROUNDS = TAX["rounds"]
NR = len(ROUNDS)


def load(name_prefix, sub):
    p = glob.glob(os.path.join(TAB, f"{name_prefix}_*_{sub}.csv")) or glob.glob(os.path.join(TAB, f"{name_prefix}*_{sub}.csv"))
    return pd.read_csv(p[0], encoding="utf-8-sig", dtype=str, keep_default_na=False)


FILES = {}
for p in glob.glob(os.path.join(TAB, "*.csv")):
    b = os.path.basename(p)[:-4]
    name, sub = b.rsplit("_", 1)
    FILES[(name, sub)] = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
NAMES = sorted({n for n, _ in FILES})
csvof = lambda name, sub: FILES[(name, sub)]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def compare(name, sub, exp, keys, tol=1e-4, order=None):
    """exp(DataFrame)와 CSV를 keys로 맞춰 모든 열을 비교한다."""
    csv = csvof(name, sub).copy()
    e = exp.copy()
    for k in keys:
        e[k] = e[k].astype(str)
    m = csv.merge(e, on=keys, how="outer", suffixes=("", "__e"), indicator=True)
    ok((m["_merge"] == "both").all(), f"{name}/{sub}: 행 집합 불일치 (csv만 {int((m['_merge']=='left_only').sum())}, 재계산만 {int((m['_merge']=='right_only').sum())})")
    mb = m[m["_merge"] == "both"]
    for c in e.columns:
        if c in keys:
            continue
        if c not in csv.columns:
            ok(False, f"{name}/{sub}: 열 없음 {c}")
            continue
        a = mb[c].map(lambda v: num(v) if v != "" else float("nan"))
        b = mb[c + "__e"]
        if b.map(lambda v: isinstance(v, str)).any():
            bad = (mb[c].astype(str) != b.astype(str).replace({"nan": "", "None": ""})).sum()
        else:
            b = b.astype(float)
            bad = (~((a - b).abs() <= tol) & ~(a.isna() & b.isna())).sum()
        ok(bad == 0, f"{name}/{sub}: 열 {c} 불일치 {int(bad)}행")
    if order is not None:
        ok(list(csv[keys[0]]) == [str(v) for v in order], f"{name}/{sub}: 정렬 순서 불일치")


def f_ans(qid):
    a = [c for c in chs[qid] if c["role"] == "정답"]
    return a[0] if len(a) == 1 else None


# ---------------------------------------------------------------- A. CSV 재계산
for sub, d in SUB.items():
    N = len(d)
    # L1
    e = pd.DataFrame({"대주제": [t["key"] for t in TAX["topics"]]})
    cnt = d.topic.value_counts()
    e["문항수"] = e.대주제.map(lambda k: int(cnt.get(k, 0)))
    e["회차당평균"] = e.문항수 / NR
    e["비중"] = 100 * e.문항수 / N
    compare("학습1_대주제별_출제비중", sub, e, ["대주제"])
    e = pd.DataFrame({"시대": chapters})
    cc = d.chapter.value_counts()
    e["문항수"] = e.시대.map(lambda k: int(cc.get(k, 0)))
    e["회차당평균"] = e.문항수 / NR
    e["비중"] = 100 * e.문항수 / N
    compare("학습1_시대별_출제비중", sub, e, ["시대"])
    # L2
    g = d.groupby("code").r.agg(["count", "nunique", "max"])
    e = pd.DataFrame({"소주제": list(subname)})
    e["출제횟수"] = e.소주제.map(lambda k: int(g["count"].get(k, 0)))
    e["출제회차수"] = e.소주제.map(lambda k: int(g["nunique"].get(k, 0)))
    e["최근출제회차"] = e.소주제.map(lambda k: g["max"].get(k, np.nan))
    e["미출제회차수"] = 79 - e["최근출제회차"]
    order = e.assign(_l=e.최근출제회차.fillna(-1)).sort_values(["출제횟수", "_l", "소주제"], ascending=[False, False, True])["소주제"]
    compare("학습2_소주제_빈출순위", sub, e, ["소주제"], order=None)
    ok(list(csvof("학습2_소주제_빈출순위", sub)["소주제"]) == list(order), f"L2/{sub}: 순위 정렬 불일치")
    # L3
    rows = []
    for _, r in d.iterrows():
        a = f_ans(r.id)
        if a:
            for kw in toks(r.kw):
                for k in toks(a["k"]):
                    rows.append((r.code, kw, k))
    e = pd.DataFrame(rows, columns=["소주제", "자료키워드", "정답키워드"]).value_counts().reset_index(name="문항수")
    compare("학습3_단서정답짝", sub, e, ["소주제", "자료키워드", "정답키워드"])
    # L4a
    wrong, right = {}, {}
    for _, r in d.iterrows():
        ws = {t for c in chs[r.id] if c["role"] == "오답" for t in toks(c["k"])}
        for t in ws:
            wrong[t] = wrong.get(t, 0) + 1
        a = f_ans(r.id)
        if a:
            for t in toks(a["k"]):
                right[t] = right.get(t, 0) + 1
    e = pd.DataFrame({"키워드": list(wrong)})
    e["오답문항수"] = e.키워드.map(wrong)
    e["정답문항수"] = e.키워드.map(lambda k: right.get(k, 0))
    e["오답비율"] = 100 * e.오답문항수 / (e.오답문항수 + e.정답문항수)
    compare("학습4_함정키워드", sub, e, ["키워드"])
    # L4b
    pr = []
    anscnt = {}
    for _, r in d.iterrows():
        a = f_ans(r.id)
        if not a or not a["code"]:
            continue
        anscnt[a["code"]] = anscnt.get(a["code"], 0) + 1
        for c in {c["code"] for c in chs[r.id] if c["role"] == "오답" and c["code"] and c["code"] != a["code"]}:
            pr.append((a["code"], c))
    e = pd.DataFrame(pr, columns=["정답소주제", "오답소주제"]).value_counts().reset_index(name="문항수")
    e["정답소주제문항수"] = e.정답소주제.map(anscnt)
    compare("학습4_함정소주제짝", sub, e, ["정답소주제", "오답소주제"])
    # L5
    rec = d[d.r >= 70]
    pri = d[d.r < 70]
    e = pd.DataFrame({"소주제": list(subname)})
    e["최근10회문항수"] = e.소주제.map(lambda k: int((rec.code == k).sum()))
    e["이전23회문항수"] = e.소주제.map(lambda k: int((pri.code == k).sum()))
    e = e[e.최근10회문항수 > 0].copy()
    e["최근회차당"] = e.최근10회문항수 / 10
    e["이전회차당"] = e.이전23회문항수 / 23
    e["배율"] = np.where(e.이전23회문항수 > 0, e.최근회차당 / e.이전회차당.replace(0, np.nan), np.nan)
    e["판정"] = np.where((e.최근10회문항수 >= 4) & ((e.이전23회문항수 == 0) | (e.배율 >= 1.5)), "집중", "")
    compare("학습5_최근10회_소주제", sub, e, ["소주제"])
    kr, kp = {}, {}
    for _, r in d.iterrows():
        for t in toks(r.kw):
            (kr if r.r >= 70 else kp)[t] = (kr if r.r >= 70 else kp).get(t, 0) + 1
    e = pd.DataFrame({"키워드": [k for k, v in kr.items() if v >= 2]})
    e["최근10회문항수"] = e.키워드.map(kr)
    e["이전23회문항수"] = e.키워드.map(lambda k: kp.get(k, 0))
    e["최근회차당"] = e.최근10회문항수 / 10
    e["이전회차당"] = e.이전23회문항수 / 23
    e["배율"] = np.where(e.이전23회문항수 > 0, e.최근회차당 / e.이전회차당.replace(0, np.nan), np.nan)
    e["판정"] = np.where((e.최근10회문항수 >= 3) & ((e.이전23회문항수 == 0) | (e.배율 >= 1.5)), "집중", "")
    compare("학습5_최근10회_키워드", sub, e, ["키워드"])
    # L6, L6b
    kwr = {}
    for _, r in d.iterrows():
        for t in toks(r.kw):
            kwr.setdefault(t, set()).add(int(r.r))

    def elig(T):
        out = []
        for t, rs in kwr.items():
            rr = sorted(x for x in rs if x < T)
            if len(rr) >= 4:
                med = float(np.median(np.diff(rr)))
                out.append((t, len(rr), rr[-1], med, T - rr[-1], (T - rr[-1]) / med))
        return out

    el = elig(80)
    e = pd.DataFrame(el, columns=["키워드", "출제회차수", "최근출제회차", "간격중앙값", "경과회차", "도래비"])
    e["후보"] = np.where(e.도래비 >= 1, "예", "아니오")
    e["최근10회출제회차수"] = e.키워드.map(lambda k: sum(1 for x in kwr[k] if x >= 70))
    compare("학습6_다음회차_후보키워드", sub, e, ["키워드"])
    tot = [0, 0, 0, 0]
    bt = csvof("학습6_후보규칙_백테스트", sub)
    for T in range(70, 80):
        actual = {t for t, rs in kwr.items() if T in rs}
        el = elig(T)
        cand = [x for x in el if x[5] >= 1]
        hit = sum(x[0] in actual for x in cand)
        top = sorted(el, key=lambda x: (-x[1], x[0]))[:len(cand)]
        th = sum(x[0] in actual for x in top)
        allh = sum(x[0] in actual for x in el)
        row = bt[bt.대상회차 == str(T)].iloc[0]
        ok((int(row.대상키워드수), int(row.후보수), int(row.후보적중), int(row.빈도상위동수적중)) == (len(el), len(cand), hit, th), f"L6b/{sub}/{T}회")
        for i, v in enumerate((len(el), len(cand), hit, allh)):
            tot[i] += v
    # 합계 행: 회차별 합산으로 다시 계산
    row = bt[bt.대상회차 == "합계"].iloc[0]
    ok((int(row.대상키워드수), int(row.후보수), int(row.후보적중)) == (tot[0], tot[1], tot[2]), f"L6b/{sub}: 합계 행")
    ok(abs(num(row.후보적중률) - 100 * tot[2] / tot[1]) < 1e-4, f"L6b/{sub}: 합계 적중률")
    # P1
    pv = pd.crosstab(d.r, d.topic).reindex(index=ROUNDS, columns=[t["key"] for t in TAX["topics"]], fill_value=0)
    e = pd.DataFrame({"대주제": pv.columns, "평균": pv.mean().values, "최소": pv.min().values, "최대": pv.max().values,
                      "표준편차": pv.std(ddof=0).values, "출제회차수": (pv > 0).sum().values})
    compare("교수1_대주제별_평균최소최대", sub, e, ["대주제"])
    pc = pd.crosstab(d.r, d.chapter).reindex(index=ROUNDS, columns=chapters, fill_value=0)
    e = pd.DataFrame({"시대": pc.columns, "평균": pc.mean().values, "최소": pc.min().values, "최대": pc.max().values,
                      "표준편차": pc.std(ddof=0).values, "출제회차수": (pc > 0).sum().values})
    compare("교수1_시대별_평균최소최대", sub, e, ["시대"])
    a1 = csvof("교수1_회차별_대주제문항수", sub)
    e = pv.reset_index().rename(columns={"r": "회차"})
    e["합계"] = pv.sum(axis=1).values
    e["회차"] = e["회차"].astype(str)
    compare("교수1_회차별_대주제문항수", sub, e, ["회차"])
    if sub == "전체":
        ok((pv.sum(axis=1) == 50).all(), "P1a: 전체 기준 회차 합계가 50이 아님")
    # P2a
    band = pd.cut(d.n, [0, 10, 20, 30, 40, 50], labels=["1~10번", "11~20번", "21~30번", "31~40번", "41~50번"])
    xt = pd.crosstab(band, d.chapter).reindex(columns=chapters, fill_value=0)
    e = xt.reset_index().rename(columns={"n": "번호대"})
    e["번호대"] = e["번호대"].astype(str)
    e["합계"] = xt.sum(axis=1).values
    compare("교수2_번호대x시대", sub, e, ["번호대"])
    # P2b
    gb = d.groupby(["r", "chapter"]).n.agg(["count", "min", "max", "median"]).reset_index()
    e = gb.rename(columns={"r": "회차", "chapter": "시대", "count": "문항수", "min": "첫번호", "max": "마지막번호", "median": "중앙번호"})
    e["회차"] = e["회차"].astype(str)
    compare("교수2_회차별_시대경계", sub, e, ["회차", "시대"])
    # P3a
    cb = d[d.blk != ""].groupby(["code", "blk"]).r.agg(["count", "nunique", "max"])
    rows = []
    for code, items in TAX["abc"].items():
        for it in items:
            k = (code, it["l"])
            rows.append((code, it["l"], int(cb["count"].get(k, 0)), int(cb["nunique"].get(k, 0)), cb["max"].get(k, np.nan)))
    e = pd.DataFrame(rows, columns=["소주제", "묶음", "문항수", "출제회차수", "최근출제회차"])
    compare("교수3_세분류묶음별_출제수", sub, e, ["소주제", "묶음"])
    ok(len(csvof("교수3_미출제_세분류묶음", sub)) == int((e.문항수 == 0).sum()), f"P3b/{sub}: 미출제 묶음 수")
    # P3d
    g = d.groupby("code").r.agg(["count", "max"])
    e = pd.DataFrame({"소주제": list(subname)})
    e["출제횟수"] = e.소주제.map(lambda k: int(g["count"].get(k, 0)))
    e["최근출제회차"] = e.소주제.map(lambda k: g["max"].get(k, np.nan))
    e["미출제회차수"] = 79 - e.최근출제회차
    compare("교수3_소주제_미출제기간", sub, e, ["소주제"])
    # P4
    tc = d.type.value_counts()
    e = pd.DataFrame({"유형": tc.index, "문항수": tc.values})
    e["비중"] = 100 * e.문항수 / N
    e["회차당평균"] = e.문항수 / NR
    e["부정형문항수"] = e.유형.map(lambda t: int(d[(d.type == t) & d.neg].shape[0]))
    e["주요시대문항수"] = e.유형.map(lambda t: int(d[d.type == t].chapter.value_counts().iloc[0]))
    compare("교수4_유형분포", sub, e, ["유형"])
    b = d.assign(b=band).groupby("b", observed=True).agg(문항수=("n", "size"), 부정형=("neg", "sum"), 세트문항=("set", lambda s: int((s != "").sum()))).reset_index()
    b["번호대"] = b.b.astype(str)
    b["부정형비율"] = 100 * b.부정형 / b.문항수
    compare("교수4_번호대별_부정형세트", sub, b.drop(columns="b"), ["번호대"])
    nb = d.groupby("n").agg(문항수=("n", "size"), 부정형=("neg", "sum"), 세트문항=("set", lambda s: int((s != "").sum()))).reindex(range(1, 51), fill_value=0).reset_index()
    nb["번호"] = nb.n
    compare("교수4_번호별_부정형세트", sub, nb.drop(columns="n"), ["번호"])
    # P5
    A, B, Cc = d[d.r <= 59], d[(d.r >= 60) & (d.r <= 69)], d[d.r >= 70]
    for fname, col, keys in (("교수5_구간별_소주제", "code", list(subname)), ("교수5_구간별_시대", "chapter", chapters)):
        e = pd.DataFrame({fname and ("소주제" if col == "code" else "시대"): keys})
        kc = e.columns[0]
        e["47~59회_회차당"] = e[kc].map(lambda k: (A[col] == k).sum() / 13)
        e["60~69회_회차당"] = e[kc].map(lambda k: (B[col] == k).sum() / 10)
        e["70~79회_회차당"] = e[kc].map(lambda k: (Cc[col] == k).sum() / 10)
        e["변화_C_A"] = e["70~79회_회차당"] - e["47~59회_회차당"]
        e["변화_B_A"] = e["60~69회_회차당"] - e["47~59회_회차당"]
        e["변화_C_B"] = e["70~79회_회차당"] - e["60~69회_회차당"]
        den = np.sqrt(e["47~59회_회차당"] / 13 + e["70~79회_회차당"] / 10)
        e["z값"] = np.where(den > 0, e["변화_C_A"] / den.replace(0, np.nan), np.nan)
        e["총문항수"] = e[kc].map(lambda k: (d[col] == k).sum())
        compare(fname, sub, e, [kc])
    e = pd.DataFrame({"유형": df.type.unique()})
    e["47~59회_회차당"] = e.유형.map(lambda k: (A.type == k).sum() / 13)
    e["70~79회_회차당"] = e.유형.map(lambda k: (Cc.type == k).sum() / 10)
    e["변화_C_A"] = e["70~79회_회차당"] - e["47~59회_회차당"]
    compare("교수5_구간별_유형", sub, e, ["유형"])
    # P6
    occ = {}
    total = 0
    for _, r in d.iterrows():
        for c in chs[r.id]:
            if c["role"] != "배열" and c["t"]:
                total += 1
                occ.setdefault(re.sub(r"\s+", " ", c["t"]).strip(), []).append((int(r.r), c["role"]))
    multi = {t: v for t, v in occ.items() if len({r for r, _ in v}) >= 2}
    e = pd.DataFrame({"선지문장": list(multi)})
    e["등장횟수"] = e.선지문장.map(lambda t: len(multi[t]))
    e["등장회차수"] = e.선지문장.map(lambda t: len({r for r, _ in multi[t]}))
    e["정답"] = e.선지문장.map(lambda t: sum(1 for _, x in multi[t] if x == "정답"))
    e["오답"] = e.선지문장.map(lambda t: sum(1 for _, x in multi[t] if x == "오답"))
    compare("교수6_재출제_선지문장", sub, e, ["선지문장"])
    s6 = csvof("교수6_재출제_요약", sub).iloc[0]
    ok((int(s6.선지수), int(s6.서로다른문장수), int(s6.둘이상회차에등장한문장수), int(s6.해당문장의선지수)) ==
       (total, len(occ), len(multi), sum(len(v) for v in multi.values())), f"P6b/{sub}")
    ok(abs(num(s6.선지중비율) - 100 * sum(len(v) for v in multi.values()) / total) < 1e-3, f"P6b/{sub}: 선지중비율")

# ---------------------------------------------------------------- B. 마크다운 표 ↔ CSV
SHORT = {"선사": "제1장 선사 문화", "고대": "제2장 한국 고대사", "고려": "제3장 고려시대사", "조선": "제4장 조선시대사",
         "근대": "제5장 한국 근대사", "일제": "제6장 일제 강점기", "현대": "제7장 한국 현대사", "복합": "제8장 복합 문항", "부록": "특별 부록"}


def parse_md(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    tables, label, i = [], None, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("**전체 1,650문항**"):
            label = "전체"
        elif ln.startswith("**status가 확인인"):
            label = "확인"
        elif ln.startswith("#"):
            label = None
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(-{3}\|)+$", lines[i + 1]):
            split = lambda s: [c.strip() for c in re.split(r"(?<!\\)\|", s.strip())[1:-1]]
            head = split(ln)
            body = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                body.append(split(lines[i]))
                i += 1
            tables.append((label, head, body, ln))
            continue
        i += 1
    return tables


def cell_ok(m, c):
    if m == "-":
        return c in ("", "nan")
    try:
        mv = float(m)
        cv = float(c)
        dec = len(m.split(".")[1]) if "." in m else 0
        return abs(mv - cv) <= 0.5 * 10 ** (-dec) + 1e-9
    except ValueError:
        return m.replace("\\|", "|") == c


def match_rows(head, body, csv):
    hc = [SHORT.get(h, h) for h in head]
    if not all(h in csv.columns for h in hc):
        return None
    idx = {}
    for j, row in enumerate(csv[hc].values.tolist()):
        idx.setdefault(row[0] if not row[0].replace(".", "").isdigit() else str(int(float(row[0]))) if False else row[0], []).append(row)
    miss = 0
    for r in body:
        cands = idx.get(r[0]) or [x for x in csv[hc].values.tolist() if cell_ok(r[0], x[0])]
        if not any(all(cell_ok(m, c) for m, c in zip(r, cand)) for cand in cands):
            miss += 1
    return miss


def merged_match(head, body):
    """'열(전체)','열(확인)' 병합 표를 각 CSV와 대조한다."""
    keys = [h for h in head if not re.search(r"\((전체|확인)\)$", h)]
    vals = [h for h in head if re.search(r"\((전체|확인)\)$", h)]
    for (name, sub0), _ in FILES.items():
        if sub0 != "전체":
            continue
        cols = set(FILES[(name, "전체")].columns)
        if not all(k in cols for k in keys) or not all(re.sub(r"\((전체|확인)\)$", "", v) in cols for v in vals):
            continue
        miss = 0
        for r in body:
            kv = dict(zip(head, r))
            for sub in ("전체", "확인"):
                csv = FILES[(name, sub)]
                m = csv
                for k in keys:
                    m = m[m[k] == kv[k]]
                if len(m) == 0:
                    miss += 1
                    continue
                row = m.iloc[0]
                for v in vals:
                    if v.endswith(f"({sub})"):
                        base = v[: -len(f"({sub})")]
                        if not cell_ok(kv[v], row[base]):
                            miss += 1
        return name, miss
    return None, None


DERIVED_HEADS = ({"구간", "평균(전체)"}, {"기준", "묶음수"}, {"소주제", "미출제 묶음"})
md_tables_checked = 0
for rep in ("학습자용.md", "교수자용.md"):
    for label, head, body, ln in parse_md(os.path.join(AN, rep)):
        md_tables_checked += 1
        if any(s <= set(head) for s in DERIVED_HEADS):
            continue
        if any(re.search(r"\((전체|확인)\)$", h) for h in head):
            name, miss = merged_match(head, body)
            ok(name is not None and miss == 0, f"{rep}: 병합 표 불일치 {head[:3]} (원천 {name}, 불일치 {miss})")
            continue
        best = None
        for (name, sub), csv in FILES.items():
            if label and sub != label:
                continue
            m = match_rows(head, body, csv)
            if m is not None and (best is None or m < best[0]):
                best = (m, name, sub)
        ok(best is not None and best[0] == 0, f"{rep}: 표가 CSV와 불일치 {head[:4]} -> {best}")

# 보고서 안의 파생 표(역행·묶음 요약·미출제 묶음)
for rep in ("교수자용.md",):
    for label, head, body, ln in parse_md(os.path.join(AN, rep)):
        if {"구간", "평균(전체)"} <= set(head):
            for r in body:
                lo, hi = [int(x) for x in re.findall(r"\d+", r[0])[:2]]
                for sub, ci in (("전체", 1), ("확인", 3)):
                    d = SUB[sub]
                    seq = []
                    vals = []
                    for rd in range(lo, hi + 1):
                        x = d[d.r == rd].sort_values("n")
                        rk = [chapters.index(c) for c in x.chapter if chapters.index(c) < 7]
                        vals.append(sum(1 for a, b in zip(rk, rk[1:]) if b < a))
                    ok(cell_ok(r[ci], str(np.mean(vals))) and cell_ok(r[ci + 1], str(max(vals))), f"역행 표 {r[0]} {sub}")
        if {"기준", "묶음수"} <= set(head):
            for r in body:
                a = csvof("교수3_세분류묶음별_출제수", r[0])
                n0 = int((a["문항수"].astype(int) == 0).sum())
                ok(int(r[1]) == len(a) and int(r[3]) == n0 and int(r[2]) == len(a) - n0 and cell_ok(r[4], str(a["문항수"].astype(int).mean())) and int(r[5]) == a["문항수"].astype(int).max(), f"묶음 요약 표 {r[0]}")
        if {"소주제", "미출제 묶음"} <= set(head):
            sub = label
            z = csvof("교수3_미출제_세분류묶음", sub)
            ok(sum(int(r[3]) for r in body) == len(z) and len(body) == z["소주제"].nunique(), f"미출제 묶음 표 {sub}")

# ---------------------------------------------------------------- C. 대시보드
html = open(os.path.join(AN, "dashboard.html"), encoding="utf-8").read()
mm = re.search(r'<script id="DATA" type="application/json">(.*?)</script>', html, re.S)
dash = json.loads(mm.group(1).replace("<\\/", "</"))
tmap = {  # 대시보드 표 id -> CSV 이름 접두
    "L1a": "학습1_대주제별_출제비중", "L1b": "학습1_시대별_출제비중", "L2": "학습2_소주제_빈출순위", "L3": "학습3_단서정답짝",
    "L4a": "학습4_함정키워드", "L4b": "학습4_함정소주제짝", "L5a": "학습5_최근10회_소주제", "L5b": "학습5_최근10회_키워드",
    "L6": "학습6_다음회차_후보키워드", "L6b": "학습6_후보규칙_백테스트", "P1b": "교수1_대주제별_평균최소최대", "P1c": "교수1_시대별_평균최소최대",
    "P2a": "교수2_번호대x시대", "P2b": "교수2_회차별_시대경계", "P2d": "교수2_회차별_시대순서역행", "P3a": "교수3_세분류묶음별_출제수",
    "P3d": "교수3_소주제_미출제기간", "P4a": "교수4_유형분포", "P4b": "교수4_유형x시대", "P4c": "교수4_번호대별_부정형세트",
    "P4d": "교수4_번호별_부정형세트", "P5a": "교수5_구간별_소주제", "P5b": "교수5_구간별_유형", "P5c": "교수5_구간별_시대",
    "P6": "교수6_재출제_선지문장", "P6b": "교수6_재출제_요약",
}
for tid, name in tmap.items():
    for sub in ("전체", "확인"):
        rows = dash[tid][sub]
        csv = csvof(name, sub)
        if tid == "L3":
            csv = csv[csv["소주제내순위"].astype(int) <= 5]
        csv = csv.iloc[: len(rows)]
        good = len(rows) == len(csv) and len(rows) > 0
        if good:
            for jr, (_, cr) in zip(rows, csv.iterrows()):
                for k, v in jr.items():
                    c = cr[k]
                    if v is None:
                        good &= c in ("", "nan")
                    elif isinstance(v, (int, float)):
                        good &= abs(float(c) - v) < 1e-5
                    else:
                        good &= str(v) == c
        ok(good, f"대시보드 {tid}/{sub}가 CSV와 다름")

# ---------------------------------------------------------------- D. 요약 5줄의 숫자
def summary_lines(path):
    t = open(path, encoding="utf-8").read().split("## 핵심 요약")[1].split("\n>")[0]
    return [l for l in t.split("\n") if re.match(r"^\d\. ", l)]


L, P = summary_lines(os.path.join(AN, "학습자용.md")), summary_lines(os.path.join(AN, "교수자용.md"))
ok(len(L) == 5 and len(P) == 5, "요약이 5줄이 아님")
d = SUB["전체"]
cc = d.chapter.value_counts().rename_axis("c").reset_index(name="n").sort_values(["n", "c"], ascending=[False, True])
t3 = cc.head(3).values.tolist()
ok(f"{t3[0][0]}" in L[0] and f"{t3[0][1]/33:.1f}문항({100*t3[0][1]/1650:.1f}%)" in L[0] and f"{t3[1][1]/33:.1f}문항" in L[0] and f"{t3[2][1]/33:.1f}문항" in L[0], "L요약1")
g = d.groupby("code").r.agg(["count", "nunique", "max"]).sort_values(["count", "max"], ascending=[False, False])
c0 = g.index[0]
zeros = [subname[k] for k in subname if k not in g.index]
ok(f"{g.iloc[0]['count']}문항이 {g.iloc[0]['nunique']}개 회차" in L[1] and f"최근 출제는 {g.iloc[0]['max']}회" in L[1] and f"174개 소주제 중 {len(g)}개" in L[1] and all(z in L[1] for z in zeros), "L요약2")
rows = []
for _, r in d.iterrows():
    a = f_ans(r.id)
    if a:
        for kw in toks(r.kw):
            for k in toks(a["k"]):
                rows.append((r.code, kw, k))
p3 = pd.DataFrame(rows, columns=["c", "kw", "k"]).value_counts().reset_index(name="n").sort_values(["n", "c", "kw", "k"], ascending=[False, True, True, True]).iloc[0]
ok(f"'{p3.kw}'→'{p3.k}'" in L[2] and f"{p3.n}문항)" in L[2], "L요약3 짝")
w4 = pd.read_csv(os.path.join(TAB, "학습4_함정키워드_전체.csv"), encoding="utf-8-sig", dtype=str).iloc[0]
ok(f"'{w4.키워드}'(오답 {w4.오답문항수}문항, 정답 {w4.정답문항수}문항)" in L[2], "L요약3 함정")
b4 = pd.read_csv(os.path.join(TAB, "학습4_함정소주제짝_전체.csv"), encoding="utf-8-sig", dtype=str).iloc[0]
ok(b4.정답소주제명 in L[2] and b4.오답소주제명 in L[2] and f"({b4.문항수}문항)" in L[2], "L요약3 소주제짝")
a5 = csvof("학습5_최근10회_소주제", "전체")
k5 = csvof("학습5_최근10회_키워드", "전체")
ok(f"소주제 {(a5.판정=='집중').sum()}개" in L[3] and f"키워드 {(k5.판정=='집중').sum()}개" in L[3] and f"({a5.iloc[0].최근10회문항수}문항, 배율 {float(a5.iloc[0].배율):.2f})" in L[3], "L요약4")
l6 = csvof("학습6_다음회차_후보키워드", "전체")
bt = csvof("학습6_후보규칙_백테스트", "전체")
bt = bt[bt.대상회차 == "합계"].iloc[0]
ok(f"후보 키워드 {(l6.후보=='예').sum()}개" in L[4] and f"{float(bt.후보적중률):.1f}%" in L[4] and f"{float(bt.빈도상위적중률):.1f}%" in L[4] and f"{float(bt.전체대상적중률):.1f}%" in L[4], "L요약5")
# 교수자용 요약
pv = pd.crosstab(d.r, d.topic).reindex(index=ROUNDS, fill_value=0)
pc = pd.crosstab(d.r, d.chapter).reindex(index=ROUNDS, columns=chapters, fill_value=0)
c4 = pc[chapters[3]]
ok(f"평균 {c4.mean():.1f}문항(최소 {c4.min()}, 최대 {c4.max()})" in P[0] and f"{pv.max().max()}문항" in P[0], "P요약1")
band = pd.cut(d.n, [0, 10, 20, 30, 40, 50], labels=["1~10번", "11~20번", "21~30번", "31~40번", "41~50번"])
xt = pd.crosstab(band, d.chapter)
okb = True
for b, row in xt.iterrows():
    top = max(chapters, key=lambda c: (row[c], -chapters.index(c)))
    okb &= f"{b} { {v: k for k, v in SHORT.items()}[top] }({100*row[top]/row.sum():.1f}%)" in P[1]
ok(okb, "P요약2 번호대")
invs = []
for rd in ROUNDS:
    x = d[d.r == rd].sort_values("n")
    rk = [chapters.index(c) for c in x.chapter if chapters.index(c) < 7]
    invs.append(sum(1 for a, b in zip(rk, rk[1:]) if b < a))
ok(f"평균 {np.mean(invs):.1f}번" in P[1], "P요약2 역행")
gb = d[d.chapter == chapters[3]].groupby("r").n.median()
pp = pd.DataFrame([{"r": r, "c": chapters[3], "m": gb.get(r)} for r in ROUNDS])
for (lo, hi) in ((47, 59), (60, 69), (70, 79)):
    ok(f"{pp[(pp.r>=lo)&(pp.r<=hi)].m.mean():.1f}번" in P[1], f"P요약2 중앙번호 {lo}-{hi}")
abc_n = sum(len(v) for v in TAX["abc"].values())
cb = d[d.blk != ""].groupby(["code", "blk"]).size()
zero = sum(1 for code, items in TAX["abc"].items() for it in items if cb.get((code, it["l"]), 0) == 0)
cbc = SUB["확인"][SUB["확인"].blk != ""].groupby(["code", "blk"]).size()
zeroc = sum(1 for code, items in TAX["abc"].items() for it in items if cbc.get((code, it["l"]), 0) == 0)
gl = d.groupby("code").r.max()
stale = gl.sort_values().index[0]
ok(f"{abc_n}개 중 {zero}개({100*zero/abc_n:.1f}%)" in P[2] and f"확인 기준 {zeroc}개" in P[2] and f"'{subname[stale]}'(마지막 출제 {gl[stale]}회, 이후 {79-gl[stale]}회" in P[2], "P요약3")
tc = d.type.value_counts()
neg = d[d.neg]
nb = neg.n.between(21, 30).sum() + neg.n.between(41, 50).sum()
sets = d[d.set != ""]
ok(f"'{tc.index[0]}'" in P[3] and f"{tc.iloc[0]:,}문항({100*tc.iloc[0]/1650:.1f}%)" in P[3] and f"부정형 {len(neg)}문항 중 {nb}문항" in P[3] and f"세트 문항 {len(sets)}문항 중 {(sets.n>=41).sum()}문항" in P[3], "P요약4")
A, Cc = d[d.r <= 59], d[d.r >= 70]
chg = {k: (Cc.code == k).sum() / 10 - (A.code == k).sum() / 13 for k in subname}
ks = sorted(chg, key=lambda k: (-chg[k], k))
kd = sorted(chg, key=lambda k: (chg[k], k))
ok(f"'{subname[ks[0]]}'({chg[ks[0]]:+.2f})" in P[4] and f"'{subname[kd[0]]}'({chg[kd[0]]:+.2f})" in P[4], "P요약5 증감")
s6 = csvof("교수6_재출제_요약", "전체").iloc[0]
ok(f"{int(s6.둘이상회차에등장한문장수):,}문장" in P[4] and f"{float(s6.선지중비율):.1f}%" in P[4], "P요약5 재출제")

# ---------------------------------------------------------------- 결과
out = [f"검증 항목 {passes + len(fails)}개 중 통과 {passes}개, 실패 {len(fails)}개 (마크다운 표 {md_tables_checked}개 대조)"] + log
print("\n".join(out))
open(os.path.join(AN, "scripts", "verify_log.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
sys.exit(1 if fails else 0)
