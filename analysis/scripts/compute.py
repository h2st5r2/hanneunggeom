"""한능검 심화 출제 경향 분석: 모든 표를 계산하는 모듈.

보고서·CSV·대시보드는 이 모듈이 돌려주는 DataFrame만 읽는다.
부분집합: '전체'(1,650문항) / '확인'(status == "확인").
"""
import json
import os
import re
import statistics
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(ROOT, "questions.json"), encoding="utf-8") as f:
    Q = json.load(f)
with open(os.path.join(ROOT, "taxonomy.json"), encoding="utf-8") as f:
    TAX = json.load(f)

ROUNDS = TAX["rounds"]              # 47..79
NR = len(ROUNDS)                    # 33
LAST = max(ROUNDS)                  # 79
NEXT = LAST + 1                     # 80
PERIODS = [("47~59회", 47, 59), ("60~69회", 60, 69), ("70~79회", 70, 79)]
RECENT_FROM = 70

TOPIC_NAME = {t["key"]: t["name"] for t in TAX["topics"]}
TOPIC_CH = {t["key"]: t["chapter"] for t in TAX["topics"]}
SUB_NAME = {s["code"]: s["name"] for t in TAX["topics"] for s in t["subs"]}
SUB_TOPIC = {s["code"]: t["key"] for t in TAX["topics"] for s in t["subs"]}
CHAPTERS = list(dict.fromkeys(t["chapter"] for t in TAX["topics"]))
ERA_CH = CHAPTERS[:7]               # 제1~7장: 연대순으로 배열되는 시대 장
CH_RANK = {c: i for i, c in enumerate(CHAPTERS)}
BANDS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
TRUE_ROLES = ("정답", "정답(부정형 옳은 선지)")

SUBSETS = {
    "전체": Q,
    "확인": [x for x in Q if x["status"] == "확인"],
}


def split_kw(s):
    """쉼표로 키워드를 나눈다. 괄호 안의 쉼표는 나누지 않는다."""
    out, cur, depth = [], "", 0
    for ch in s or "":
        if ch in "([（":
            depth += 1
        elif ch in ")]）":
            depth = max(0, depth - 1)
        if ch in ",，" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    res = []
    for t in out:
        t = re.sub(r"\s+", " ", t).strip()
        if t and t not in res:
            res.append(t)
    return res


def ans_choice(x):
    a = [c for c in x["ch"] if c["role"] == "정답"]
    return a[0] if len(a) == 1 else None


def chapter_of(x):
    return TOPIC_CH[x["topic"]]


def band_of(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return f"{lo}~{hi}번"


def period_of(r):
    for name, lo, hi in PERIODS:
        if lo <= r <= hi:
            return name


def ordinal(df):
    df = df.reset_index(drop=True)
    df.insert(0, "순위", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- 학습자용

def t_L1a(qs):
    c = Counter(x["topic"] for x in qs)
    N = len(qs)
    rows = [dict(대주제=t["key"], 대주제명=t["name"], 시대=t["chapter"], 문항수=c.get(t["key"], 0),
                 회차당평균=c.get(t["key"], 0) / NR, 비중=100 * c.get(t["key"], 0) / N)
            for t in TAX["topics"]]
    return pd.DataFrame(rows)


def t_L1b(qs):
    c = Counter(chapter_of(x) for x in qs)
    N = len(qs)
    return pd.DataFrame([dict(시대=ch, 문항수=c.get(ch, 0), 회차당평균=c.get(ch, 0) / NR,
                              비중=100 * c.get(ch, 0) / N) for ch in CHAPTERS])


def t_L2(qs):
    by = defaultdict(list)
    for x in qs:
        by[x["code"]].append(x["r"])
    rows = []
    for code, name in SUB_NAME.items():
        rs = by.get(code, [])
        rows.append(dict(소주제=code, 소주제명=name, 대주제=SUB_TOPIC[code], 출제횟수=len(rs),
                         출제회차수=len(set(rs)), 최근출제회차=max(rs) if rs else None,
                         미출제회차수=(LAST - max(rs)) if rs else None))
    df = pd.DataFrame(rows)
    df["_l"] = df["최근출제회차"].fillna(-1)
    df = df.sort_values(["출제횟수", "_l", "소주제"], ascending=[False, False, True]).drop(columns="_l")
    return ordinal(df)


def t_L3(qs):
    pairs, qn = Counter(), Counter()
    for x in qs:
        a = ans_choice(x)
        if not a:
            continue
        qn[x["code"]] += 1
        for kw in split_kw(x["kw"]):
            for k in split_kw(a["k"]):
                pairs[(x["code"], kw, k)] += 1
    rows = [dict(소주제=c, 소주제명=SUB_NAME[c], 자료키워드=kw, 정답키워드=k, 문항수=n, 소주제정답문항수=qn[c])
            for (c, kw, k), n in pairs.items()]
    df = pd.DataFrame(rows).sort_values(["소주제", "문항수", "자료키워드", "정답키워드"],
                                        ascending=[True, False, True, True])
    df["소주제내순위"] = df.groupby("소주제").cumcount() + 1
    return df.reset_index(drop=True)


def t_L4a(qs):
    wrong, right, wcode = defaultdict(set), defaultdict(set), defaultdict(Counter)
    for x in qs:
        for c in x["ch"]:
            if c["role"] == "오답":
                for k in split_kw(c["k"]):
                    wrong[k].add(x["id"])
                    if c["code"]:
                        wcode[k][c["code"]] += 1
        a = ans_choice(x)
        if a:
            for k in split_kw(a["k"]):
                right[k].add(x["id"])
    rows = []
    for k, ids in wrong.items():
        w, r = len(ids), len(right.get(k, ()))
        top = sorted(wcode[k].items(), key=lambda kv: (-kv[1], kv[0]))
        code = top[0][0] if top else None
        rows.append(dict(키워드=k, 오답문항수=w, 정답문항수=r, 오답비율=100 * w / (w + r),
                         주소주제=code, 소주제명=SUB_NAME.get(code)))
    df = pd.DataFrame(rows).sort_values(["오답문항수", "오답비율", "키워드"], ascending=[False, False, True])
    return ordinal(df)


def t_L4b(qs):
    pairs, an = Counter(), Counter()
    for x in qs:
        a = ans_choice(x)
        if not a or not a["code"]:
            continue
        an[a["code"]] += 1
        for d in {c["code"] for c in x["ch"] if c["role"] == "오답" and c["code"] and c["code"] != a["code"]}:
            pairs[(a["code"], d)] += 1
    rows = [dict(정답소주제=a, 정답소주제명=SUB_NAME[a], 오답소주제=d, 오답소주제명=SUB_NAME[d],
                 문항수=n, 정답소주제문항수=an[a], 같은대주제=("예" if SUB_TOPIC[a] == SUB_TOPIC[d] else "아니오"))
            for (a, d), n in pairs.items()]
    df = pd.DataFrame(rows).sort_values(["문항수", "정답소주제", "오답소주제"], ascending=[False, True, True])
    return ordinal(df)


def self_pair_count(qs):
    """오답 선지가 정답 선지와 같은 소주제인 문항 수 (L4b에서 제외한 짝)."""
    n = 0
    for x in qs:
        a = ans_choice(x)
        if a and a["code"] and any(c["role"] == "오답" and c["code"] == a["code"] for c in x["ch"]):
            n += 1
    return n


def _lift(recent, prior, n_recent, n_prior):
    if prior == 0:
        return None
    return (recent / n_recent) / (prior / n_prior)


def t_L5a(qs):
    n_rec = sum(1 for r in ROUNDS if r >= RECENT_FROM)
    n_pri = NR - n_rec
    rec, pri = Counter(), Counter()
    for x in qs:
        (rec if x["r"] >= RECENT_FROM else pri)[x["code"]] += 1
    rows = []
    for c, name in SUB_NAME.items():
        a, b = rec.get(c, 0), pri.get(c, 0)
        lift = _lift(a, b, n_rec, n_pri)
        flag = "집중" if a >= 4 and (b == 0 or lift >= 1.5) else ""
        rows.append(dict(소주제=c, 소주제명=name, 최근10회문항수=a, 이전23회문항수=b,
                         최근회차당=a / n_rec, 이전회차당=b / n_pri, 배율=lift, 판정=flag))
    df = pd.DataFrame(rows)
    df["_b"] = df["배율"].fillna(1e9)
    df = df.sort_values(["최근10회문항수", "_b", "소주제"], ascending=[False, False, True]).drop(columns="_b")
    return ordinal(df[df["최근10회문항수"] > 0])


def t_L5b(qs):
    n_rec = sum(1 for r in ROUNDS if r >= RECENT_FROM)
    n_pri = NR - n_rec
    rec, pri = Counter(), Counter()
    for x in qs:
        for k in split_kw(x["kw"]):
            (rec if x["r"] >= RECENT_FROM else pri)[k] += 1
    rows = []
    for k, a in rec.items():
        if a < 2:
            continue
        b = pri.get(k, 0)
        lift = _lift(a, b, n_rec, n_pri)
        flag = "집중" if a >= 3 and (b == 0 or lift >= 1.5) else ""
        rows.append(dict(키워드=k, 최근10회문항수=a, 이전23회문항수=b, 최근회차당=a / n_rec,
                         이전회차당=b / n_pri, 배율=lift, 판정=flag))
    df = pd.DataFrame(rows)
    df["_b"] = df["배율"].fillna(1e9)
    df = df.sort_values(["최근10회문항수", "_b", "키워드"], ascending=[False, False, True]).drop(columns="_b")
    return ordinal(df)


def _kw_rounds(qs, before):
    d = defaultdict(set)
    for x in qs:
        if x["r"] < before:
            for k in split_kw(x["kw"]):
                d[k].add(x["r"])
    return d


def _kw_stats(rs, T):
    rs = sorted(rs)
    gaps = [b - a for a, b in zip(rs, rs[1:])]
    med = statistics.median(gaps)
    last = rs[-1]
    el = T - last
    return len(rs), last, med, el, el / med


def t_L6(qs):
    d = _kw_rounds(qs, NEXT)
    rows = []
    for k, rs in d.items():
        if len(rs) < 4:
            continue
        n, last, med, el, due = _kw_stats(rs, NEXT)
        rows.append(dict(키워드=k, 출제회차수=n, 최근출제회차=last, 간격중앙값=med, 경과회차=el, 도래비=due,
                         최근10회출제회차수=sum(1 for r in rs if r >= RECENT_FROM),
                         후보="예" if due >= 1 else "아니오"))
    df = pd.DataFrame(rows)
    df["_c"] = (df["후보"] == "예").astype(int)
    df = df.sort_values(["_c", "출제회차수", "도래비", "키워드"], ascending=[False, False, False, True]).drop(columns="_c")
    return ordinal(df)


def t_L6b(qs):
    rows = []
    th = tn = tb = tr = 0
    for T in range(RECENT_FROM, NEXT):
        d = _kw_rounds(qs, T)
        actual = {k for x in qs if x["r"] == T for k in split_kw(x["kw"])}
        el = []
        for k, rs in d.items():
            if len(rs) >= 4:
                n, last, med, e, due = _kw_stats(rs, T)
                el.append((k, n, due))
        cand = [e for e in el if e[2] >= 1]
        N = len(cand)
        hits = sum(1 for e in cand if e[0] in actual)
        top = sorted(el, key=lambda e: (-e[1], e[0]))[:N]
        bh = sum(1 for e in top if e[0] in actual)
        rh = sum(1 for e in el if e[0] in actual)
        rows.append(dict(대상회차=T, 대상키워드수=len(el), 후보수=N, 후보적중=hits,
                         후보적중률=100 * hits / N if N else None,
                         빈도상위동수적중=bh, 빈도상위적중률=100 * bh / N if N else None,
                         전체대상적중률=100 * rh / len(el) if el else None))
        th += hits; tn += N; tb += bh; tr += rh
    ne = sum(r["대상키워드수"] for r in rows)
    rows.append(dict(대상회차="합계", 대상키워드수=ne, 후보수=tn, 후보적중=th,
                     후보적중률=100 * th / tn, 빈도상위동수적중=tb, 빈도상위적중률=100 * tb / tn,
                     전체대상적중률=100 * tr / ne))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 교수자용

def _round_topic_counts(qs):
    c = Counter((x["r"], x["topic"]) for x in qs)
    return c


def t_P1a(qs):
    c = _round_topic_counts(qs)
    keys = [t["key"] for t in TAX["topics"]]
    rows = []
    for r in ROUNDS:
        row = {"회차": r}
        for k in keys:
            row[k] = c.get((r, k), 0)
        row["합계"] = sum(row[k] for k in keys)
        rows.append(row)
    return pd.DataFrame(rows)


def _summ(vals):
    return dict(평균=sum(vals) / len(vals), 최소=min(vals), 최대=max(vals),
                표준편차=statistics.pstdev(vals), 출제회차수=sum(1 for v in vals if v > 0))


def t_P1b(qs):
    c = _round_topic_counts(qs)
    rows = []
    for t in TAX["topics"]:
        vals = [c.get((r, t["key"]), 0) for r in ROUNDS]
        rows.append(dict(대주제=t["key"], 대주제명=t["name"], 시대=t["chapter"], **_summ(vals)))
    return pd.DataFrame(rows)


def t_P1c(qs):
    c = Counter((x["r"], chapter_of(x)) for x in qs)
    rows = []
    for ch in CHAPTERS:
        vals = [c.get((r, ch), 0) for r in ROUNDS]
        rows.append(dict(시대=ch, **_summ(vals)))
    return pd.DataFrame(rows)


def t_P2a(qs):
    c = Counter((band_of(x["n"]), chapter_of(x)) for x in qs)
    rows = []
    for lo, hi in BANDS:
        b = f"{lo}~{hi}번"
        row = {"번호대": b}
        for ch in CHAPTERS:
            row[ch] = c.get((b, ch), 0)
        row["합계"] = sum(row[ch] for ch in CHAPTERS)
        rows.append(row)
    return pd.DataFrame(rows)


def t_P2b(qs):
    by = defaultdict(list)
    for x in qs:
        by[(x["r"], chapter_of(x))].append(x["n"])
    rows = []
    for r in ROUNDS:
        for ch in CHAPTERS:
            ns = by.get((r, ch))
            if ns:
                rows.append(dict(회차=r, 시대=ch, 문항수=len(ns), 첫번호=min(ns), 마지막번호=max(ns),
                                 중앙번호=statistics.median(ns)))
    return pd.DataFrame(rows)


def t_P2c(qs):
    b = t_P2b(qs)
    rows = []
    for name, lo, hi in PERIODS:
        for ch in CHAPTERS:
            s = b[(b["회차"] >= lo) & (b["회차"] <= hi) & (b["시대"] == ch)]
            if len(s):
                rows.append(dict(구간=name, 시대=ch, 등장회차수=len(s), 회차당문항수=s["문항수"].sum() / (hi - lo + 1),
                                 첫번호평균=s["첫번호"].mean(), 마지막번호평균=s["마지막번호"].mean(),
                                 중앙번호평균=s["중앙번호"].mean()))
    return pd.DataFrame(rows)


def t_P2d(qs):
    rows = []
    for r in ROUNDS:
        seq = [CH_RANK[chapter_of(x)] for x in sorted((y for y in qs if y["r"] == r), key=lambda y: y["n"])
               if chapter_of(x) in ERA_CH]
        inv = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
        rows.append(dict(회차=r, 시대장문항수=len(seq), 역행횟수=inv))
    return pd.DataFrame(rows)


def t_P3a(qs):
    cnt, rs, = Counter(), defaultdict(set)
    for x in qs:
        b = x.get("blk")
        if b:
            cnt[(x["code"], b)] += 1
            rs[(x["code"], b)].add(x["r"])
    rows = []
    for code, items in TAX["abc"].items():
        for it in items:
            k = (code, it["l"])
            rows.append(dict(소주제=code, 소주제명=SUB_NAME.get(code), 묶음=it["l"], 묶음명=it["name"],
                             문항수=cnt.get(k, 0), 출제회차수=len(rs.get(k, ())),
                             최근출제회차=max(rs[k]) if k in rs else None))
    return pd.DataFrame(rows)


def t_P3b(qs):
    df = t_P3a(qs)
    df = df[df["문항수"] == 0].copy()
    # 같은 소주제의 다른 묶음 출제 수를 함께 보여 준다
    tot = t_P3a(qs).groupby("소주제")["문항수"].sum()
    df["소주제내_다른묶음문항수"] = df["소주제"].map(tot)
    return df.reset_index(drop=True)


def t_P3c(qs):
    c = Counter()
    for x in qs:
        b = x.get("blk")
        has = x["code"] in TAX["abc"]
        if not b:
            c[("blk 없음/빈 값", "세분류 있는 소주제" if has else "세분류 없는 소주제")] += 1
        elif x.get("blkAt"):
            c[("blkAt 표시(묶음에는 포함)", "세분류 있는 소주제" if has else "세분류 없는 소주제")] += 1
        else:
            c[("묶음 배정됨", "세분류 있는 소주제" if has else "세분류 없는 소주제")] += 1
    return pd.DataFrame([dict(구분=a, 소주제유형=b, 문항수=n) for (a, b), n in sorted(c.items())])


def t_P3d(qs):
    by = defaultdict(list)
    for x in qs:
        by[x["code"]].append(x["r"])
    rows = []
    for code, name in SUB_NAME.items():
        rs = by.get(code, [])
        rows.append(dict(소주제=code, 소주제명=name, 대주제=SUB_TOPIC[code], 출제횟수=len(rs),
                         최근출제회차=max(rs) if rs else None,
                         미출제회차수=(LAST - max(rs)) if rs else None))
    df = pd.DataFrame(rows)
    df["_g"] = df["미출제회차수"].fillna(999)
    df = df.sort_values(["_g", "소주제"], ascending=[False, True]).drop(columns="_g")
    return ordinal(df)


def t_P4a(qs):
    N = len(qs)
    rows = []
    c = Counter(x["type"] for x in qs)
    for tp, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        sub = [x for x in qs if x["type"] == tp]
        ch = Counter(chapter_of(x) for x in sub).most_common()
        top_ch = sorted(Counter(chapter_of(x) for x in sub).items(), key=lambda kv: (-kv[1], CH_RANK[kv[0]]))[0]
        top_tp = sorted(Counter(x["topic"] for x in sub).items(), key=lambda kv: (-kv[1], kv[0]))[0]
        rows.append(dict(유형=tp, 문항수=n, 비중=100 * n / N, 회차당평균=n / NR,
                         부정형문항수=sum(1 for x in sub if x["neg"]),
                         주요시대=top_ch[0], 주요시대문항수=top_ch[1], 주요시대내비중=100 * top_ch[1] / n,
                         주요대주제=f"{top_tp[0]} {TOPIC_NAME[top_tp[0]]}", 주요대주제문항수=top_tp[1]))
    return pd.DataFrame(rows)


def t_P4b(qs):
    c = Counter((x["type"], chapter_of(x)) for x in qs)
    types = [t for t, _ in sorted(Counter(x["type"] for x in qs).items(), key=lambda kv: (-kv[1], kv[0]))]
    rows = []
    for tp in types:
        row = {"유형": tp}
        for ch in CHAPTERS:
            row[ch] = c.get((tp, ch), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def t_P4c(qs):
    rows = []
    for lo, hi in BANDS:
        sub = [x for x in qs if lo <= x["n"] <= hi]
        neg = sum(1 for x in sub if x["neg"])
        rows.append(dict(번호대=f"{lo}~{hi}번", 문항수=len(sub), 부정형=neg,
                         부정형비율=100 * neg / len(sub) if sub else None,
                         세트문항=sum(1 for x in sub if x["set"])))
    return pd.DataFrame(rows)


def t_P4d(qs):
    rows = []
    for n in range(1, 51):
        sub = [x for x in qs if x["n"] == n]
        rows.append(dict(번호=n, 문항수=len(sub), 부정형=sum(1 for x in sub if x["neg"]),
                         세트문항=sum(1 for x in sub if x["set"])))
    return pd.DataFrame(rows)


def t_P4e(qs):
    by = defaultdict(list)
    for x in qs:
        if x["set"]:
            by[x["set"]].append(x)
    rows = []
    for s, xs in sorted(by.items()):
        rows.append(dict(세트=s, 문항수=len(xs), 회차=", ".join(str(r) for r in sorted({x["r"] for x in xs})),
                         유형=", ".join(sorted({x["type"] for x in xs}))))
    return pd.DataFrame(rows)


def _period_table(qs, keyfn, labelfn, keys):
    cnt = Counter((period_of(x["r"]), keyfn(x)) for x in qs)
    rows = []
    for k in keys:
        rec = {}
        for name, lo, hi in PERIODS:
            rec[name] = cnt.get((name, k), 0) / (hi - lo + 1)
        nA, nC = PERIODS[0][2] - PERIODS[0][1] + 1, PERIODS[2][2] - PERIODS[2][1] + 1
        a, b, c = rec["47~59회"], rec["60~69회"], rec["70~79회"]
        den = (a / nA + c / nC) ** 0.5
        z = (c - a) / den if den > 0 else None
        rows.append(dict(**labelfn(k), **{f"{n}_회차당": rec[n] for n, _, _ in PERIODS},
                         변화_B_A=b - a, 변화_C_B=c - b, 변화_C_A=c - a, z값=z,
                         총문항수=sum(cnt.get((n, k), 0) for n, _, _ in PERIODS)))
    df = pd.DataFrame(rows)
    df["_a"] = df["변화_C_A"].abs()
    return df


def t_P5a(qs):
    df = _period_table(qs, lambda x: x["code"], lambda k: dict(소주제=k, 소주제명=SUB_NAME[k]), list(SUB_NAME))
    return df.sort_values(["_a", "소주제"], ascending=[False, True]).drop(columns="_a").reset_index(drop=True)


def t_P5b(qs):
    types = sorted({x["type"] for x in Q}, key=lambda t: (-Counter(x["type"] for x in Q)[t], t))
    df = _period_table(qs, lambda x: x["type"], lambda k: dict(유형=k), types)
    return df.sort_values(["_a", "유형"], ascending=[False, True]).drop(columns="_a").reset_index(drop=True)


def t_P5c(qs):
    df = _period_table(qs, chapter_of, lambda k: dict(시대=k), CHAPTERS)
    return df.drop(columns="_a").reset_index(drop=True)


def _norm(t):
    return re.sub(r"\s+", " ", t).strip()


def t_P6(qs):
    d = defaultdict(list)
    for x in qs:
        for c in x["ch"]:
            if c["role"] != "배열" and c["t"]:
                d[_norm(c["t"])].append((x["r"], x["id"], c["role"]))
    rows = []
    for t, occ in d.items():
        rs = sorted({r for r, _, _ in occ})
        if len(rs) < 2:
            continue
        rc = Counter(role for _, _, role in occ)
        rows.append(dict(선지문장=t, 등장횟수=len(occ), 등장회차수=len(rs),
                         회차=", ".join(map(str, rs)),
                         정답=rc.get("정답", 0), 오답=rc.get("오답", 0),
                         부정형_옳은선지=rc.get("정답(부정형 옳은 선지)", 0),
                         부정형_정답=rc.get("오답(부정형 정답)", 0),
                         역할혼재="예" if (sum(rc.get(r, 0) for r in TRUE_ROLES) > 0 and
                                         rc.get("오답", 0) + rc.get("오답(부정형 정답)", 0) > 0) else "아니오"))
    df = pd.DataFrame(rows).sort_values(["등장회차수", "등장횟수", "선지문장"], ascending=[False, False, True])
    return ordinal(df)


def t_P6b(qs):
    d = defaultdict(set)
    total = 0
    for x in qs:
        for c in x["ch"]:
            if c["role"] != "배열" and c["t"]:
                total += 1
                d[_norm(c["t"])].add(x["r"])
    multi = [t for t, rs in d.items() if len(rs) >= 2]
    inv = 0
    occ = defaultdict(int)
    for x in qs:
        for c in x["ch"]:
            if c["role"] != "배열" and c["t"]:
                occ[_norm(c["t"])] += 1
    inv = sum(occ[t] for t in multi)
    return pd.DataFrame([dict(선지수=total, 서로다른문장수=len(d), 둘이상회차에등장한문장수=len(multi),
                              해당문장의선지수=inv, 선지중비율=100 * inv / total,
                              문장중비율=100 * len(multi) / len(d))])


# ---------------------------------------------------------------- 시대별·소주제별

def _sub_chapter(code):
    return TOPIC_CH[SUB_TOPIC[code]]


def t_E1(qs):
    """시대 안 소주제 비중·출제 주기·주요 유형."""
    by = defaultdict(list)
    for x in qs:
        by[x["code"]].append(x)
    ch_total = Counter(chapter_of(x) for x in qs)
    rows = []
    for code, name in SUB_NAME.items():
        xs = by.get(code, [])
        ch = _sub_chapter(code)
        rs = sorted({x["r"] for x in xs})
        gaps = [b - a for a, b in zip(rs, rs[1:])]
        tc = sorted(Counter(x["type"] for x in xs).items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append(dict(시대=ch, 소주제=code, 소주제명=name, 문항수=len(xs),
                         시대내비중=(100 * len(xs) / ch_total[ch]) if ch_total[ch] else None,
                         출제회차수=len(rs), 최근출제회차=rs[-1] if rs else None,
                         간격중앙값=statistics.median(gaps) if gaps else None,
                         미출제회차수=(LAST - rs[-1]) if rs else None,
                         주요유형=tc[0][0] if tc else None, 주요유형문항수=tc[0][1] if tc else None,
                         표본="표본 부족" if len(xs) < 5 else ""))
    df = pd.DataFrame(rows)
    df["_c"] = df["시대"].map(CH_RANK)
    df["_l"] = df["최근출제회차"].fillna(-1)
    df = df.sort_values(["_c", "문항수", "_l", "소주제"], ascending=[True, False, False, True]).drop(columns=["_c", "_l"])
    df.insert(2, "시대내순위", df.groupby("시대").cumcount() + 1)
    return df.reset_index(drop=True)


def t_E2(qs):
    """소주제 카드: 자료·정답·함정 키워드 상위 5개."""
    kinds = {"자료키워드": defaultdict(Counter), "정답키워드": defaultdict(Counter), "함정키워드": defaultdict(Counter)}
    for x in qs:
        for k in split_kw(x["kw"]):
            kinds["자료키워드"][x["code"]][k] += 1
        a = ans_choice(x)
        if a:
            for k in split_kw(a["k"]):
                kinds["정답키워드"][x["code"]][k] += 1
        for k in {t for c in x["ch"] if c["role"] == "오답" for t in split_kw(c["k"])}:
            kinds["함정키워드"][x["code"]][k] += 1
    rows = []
    for kind, d in kinds.items():
        for code, cnt in d.items():
            for i, (k, n) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:5], 1):
                rows.append(dict(시대=_sub_chapter(code), 소주제=code, 소주제명=SUB_NAME[code], 구분=kind, 순위=i, 키워드=k, 문항수=n))
    df = pd.DataFrame(rows)
    df["_c"] = df["시대"].map(CH_RANK)
    df["_k"] = df["구분"].map({"자료키워드": 0, "정답키워드": 1, "함정키워드": 2})
    return df.sort_values(["_c", "소주제", "_k", "순위"]).drop(columns=["_c", "_k"]).reset_index(drop=True)


def t_E3(qs):
    """정답 소주제별 함께 오답으로 묶이는 소주제 상위 3개."""
    df = t_L4b(qs)
    df = df.sort_values(["정답소주제", "문항수", "오답소주제"], ascending=[True, False, True])
    df["순위"] = df.groupby("정답소주제").cumcount() + 1
    df = df[df["순위"] <= 3].copy()
    df.insert(0, "시대", df["정답소주제"].map(_sub_chapter))
    df["_c"] = df["시대"].map(CH_RANK)
    df = df.sort_values(["_c", "정답소주제", "순위"]).drop(columns="_c")
    return df[["시대", "정답소주제", "정답소주제명", "순위", "오답소주제", "오답소주제명", "문항수", "정답소주제문항수", "같은대주제"]].reset_index(drop=True)


def t_E4(qs):
    """세분류 묶음별 출제 수와 소주제 안 비중."""
    df = t_P3a(qs)
    tot = df.groupby("소주제")["문항수"].transform("sum")
    df["소주제내비중"] = (100 * df["문항수"] / tot).where(tot > 0)
    df.insert(0, "시대", df["소주제"].map(_sub_chapter))
    return df[["시대", "소주제", "소주제명", "묶음", "묶음명", "문항수", "소주제내비중", "출제회차수", "최근출제회차"]].reset_index(drop=True)


def _top(counter, n=5):
    return "; ".join(f"{k}({v})" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def t_E5(qs):
    rows = []
    for ch in CHAPTERS:
        xs = [x for x in qs if chapter_of(x) == ch]
        sub, kw, ak, tr, ty = Counter(), Counter(), Counter(), Counter(), Counter()
        for x in xs:
            sub[SUB_NAME[x["code"]]] += 1
            ty[x["type"]] += 1
            kw.update(split_kw(x["kw"]))
            a = ans_choice(x)
            if a:
                ak.update(split_kw(a["k"]))
            tr.update({t for c in x["ch"] if c["role"] == "오답" for t in split_kw(c["k"])})
        subs = [s["code"] for t in TAX["topics"] if t["chapter"] == ch for s in t["subs"]]
        rows.append(dict(시대=ch, 문항수=len(xs), 출제소주제수=len({x["code"] for x in xs}), 분류표소주제수=len(subs),
                         소주제TOP5=_top(sub), 자료키워드TOP5=_top(kw), 정답키워드TOP5=_top(ak),
                         함정키워드TOP5=_top(tr), 주요유형TOP3=_top(ty, 3)))
    return pd.DataFrame(rows)


def t_E6(qs):
    df = t_P5a(qs)
    df.insert(2, "시대", df["소주제"].map(_sub_chapter))
    df["_c"] = df["시대"].map(CH_RANK)
    df = df.sort_values(["_c", "변화_C_A", "소주제"], ascending=[True, False, True]).drop(columns="_c")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 출제 단서(시각화용)
ALL_TYPES = [t for t, _ in sorted(Counter(x["type"] for x in Q).items(), key=lambda kv: (-kv[1], kv[0]))]


def t_Q1(qs):
    c = Counter((x["code"], x["r"]) for x in qs)
    rows = []
    for code, name in SUB_NAME.items():
        row = dict(시대=_sub_chapter(code), 소주제=code, 소주제명=name)
        for r in ROUNDS:
            row[str(r)] = c.get((code, r), 0)
        row["합계"] = sum(row[str(r)] for r in ROUNDS)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_c"] = df["시대"].map(CH_RANK)
    return df.sort_values(["_c", "소주제"]).drop(columns="_c").reset_index(drop=True)


def t_Q2(qs):
    c = Counter((x["r"], chapter_of(x)) for x in qs)
    rows = []
    for r in ROUNDS:
        row = {"회차": r}
        for ch in CHAPTERS:
            row[ch] = c.get((r, ch), 0)
        row["합계"] = sum(row[ch] for ch in CHAPTERS)
        rows.append(row)
    return pd.DataFrame(rows)


def t_Q3(qs):
    c = Counter((x["r"], x["type"]) for x in qs)
    rows = []
    for r in ROUNDS:
        row = {"회차": r}
        for t in ALL_TYPES:
            row[t] = c.get((r, t), 0)
        row["합계"] = sum(row[t] for t in ALL_TYPES)
        rows.append(row)
    return pd.DataFrame(rows)


def t_Q4(qs):
    d = _kw_rounds(qs, NEXT)
    rows = []
    for k, rs in d.items():
        if len(rs) < 4:
            continue
        n, last, med, el, due = _kw_stats(rs, NEXT)
        srt = sorted(rs)
        rows.append(dict(키워드=k, 출제회차수=n, 최근출제회차=last, 간격중앙값=med, 경과회차=el, 도래비=due,
                         마지막간격=srt[-1] - srt[-2], 회차=", ".join(map(str, srt))))
    df = pd.DataFrame(rows)
    return df.sort_values(["출제회차수", "도래비", "키워드"], ascending=[False, False, True]).reset_index(drop=True)


def _bucket(g):
    return str(g) if g <= 10 else "11+"


def t_Q5(qs):
    ent = {"소주제": defaultdict(set), "키워드": defaultdict(set), "선지문장": defaultdict(set)}
    for x in qs:
        ent["소주제"][x["code"]].add(x["r"])
        for k in split_kw(x["kw"]):
            ent["키워드"][k].add(x["r"])
        for c in x["ch"]:
            if c["role"] != "배열" and c["t"]:
                ent["선지문장"][_norm(c["t"])].add(x["r"])
    labels = [str(i) for i in range(1, 11)] + ["11+"]
    rows = []
    for kind, d in ent.items():
        cnt = Counter()
        for rs in d.values():
            srt = sorted(rs)
            for a, b in zip(srt, srt[1:]):
                cnt[_bucket(b - a)] += 1
        tot = sum(cnt.values())
        for lb in labels:
            rows.append(dict(대상=kind, 간격=lb, 건수=cnt.get(lb, 0), 비중=100 * cnt.get(lb, 0) / tot if tot else None))
    return pd.DataFrame(rows)


# (id, 파일명, 제목, 함수)
TABLES = [
    ("L1a", "학습1_대주제별_출제비중", t_L1a), ("L1b", "학습1_시대별_출제비중", t_L1b),
    ("L2", "학습2_소주제_빈출순위", t_L2),
    ("L3", "학습3_단서정답짝", t_L3),
    ("L4a", "학습4_함정키워드", t_L4a), ("L4b", "학습4_함정소주제짝", t_L4b),
    ("L5a", "학습5_최근10회_소주제", t_L5a), ("L5b", "학습5_최근10회_키워드", t_L5b),
    ("L6", "학습6_다음회차_후보키워드", t_L6), ("L6b", "학습6_후보규칙_백테스트", t_L6b),
    ("P1a", "교수1_회차별_대주제문항수", t_P1a), ("P1b", "교수1_대주제별_평균최소최대", t_P1b),
    ("P1c", "교수1_시대별_평균최소최대", t_P1c),
    ("P2a", "교수2_번호대x시대", t_P2a), ("P2b", "교수2_회차별_시대경계", t_P2b),
    ("P2c", "교수2_구간별_시대경계", t_P2c), ("P2d", "교수2_회차별_시대순서역행", t_P2d),
    ("P3a", "교수3_세분류묶음별_출제수", t_P3a), ("P3b", "교수3_미출제_세분류묶음", t_P3b),
    ("P3c", "교수3_세분류_미배정현황", t_P3c), ("P3d", "교수3_소주제_미출제기간", t_P3d),
    ("P4a", "교수4_유형분포", t_P4a), ("P4b", "교수4_유형x시대", t_P4b),
    ("P4c", "교수4_번호대별_부정형세트", t_P4c), ("P4d", "교수4_번호별_부정형세트", t_P4d),
    ("P4e", "교수4_세트문항", t_P4e),
    ("P5a", "교수5_구간별_소주제", t_P5a), ("P5b", "교수5_구간별_유형", t_P5b),
    ("P5c", "교수5_구간별_시대", t_P5c),
    ("P6", "교수6_재출제_선지문장", t_P6), ("P6b", "교수6_재출제_요약", t_P6b),
    ("E1", "시대1_소주제_비중과주기", t_E1), ("E2", "시대2_소주제카드_키워드", t_E2),
    ("E3", "시대3_소주제카드_함정짝", t_E3), ("E4", "시대4_세분류_비중", t_E4),
    ("E5", "시대5_시대요약", t_E5), ("E6", "시대6_소주제_구간변화", t_E6),
    ("Q1", "단서1_소주제_회차이력", t_Q1), ("Q2", "단서2_회차별_시대배분", t_Q2),
    ("Q3", "단서3_회차별_유형배분", t_Q3), ("Q4", "단서4_키워드_출제이력", t_Q4),
    ("Q5", "단서5_재출제간격분포", t_Q5),
]


def compute_all():
    return {tid: {sub: fn(qs) for sub, qs in SUBSETS.items()} for tid, _, fn in TABLES}


if __name__ == "__main__":
    res = compute_all()
    for tid, name, _ in TABLES:
        print("=====", tid, name, {s: res[tid][s].shape for s in SUBSETS})
