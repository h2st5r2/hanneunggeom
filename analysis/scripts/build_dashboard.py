"""analysis/dashboard.html 생성: 계산된 표를 JSON으로 내장한 단일 HTML 파일."""
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute as C  # noqa: E402

OUT = os.path.join(C.ROOT, "analysis")
R = C.compute_all()
SUBS = list(C.SUBSETS)

# 대시보드에 내장하는 표와 행 수 제한 (None = 전체). 행 순서는 표의 순서를 따른다.
EMBED = {
    "L1a": None, "L1b": None, "L2": None,
    "L3": "rank5", "L4a": 30, "L4b": 25, "L5a": None, "L5b": 40,
    "L6": None, "L6b": None,
    "P1b": None, "P1c": None, "P2a": None, "P2b": None, "P2d": None,
    "P3a": None, "P3d": 30, "P4a": None, "P4b": None, "P4c": None, "P4d": None,
    "P5a": None, "P5b": None, "P5c": None, "P6": 40, "P6b": None,
    "E1": None, "E2": None, "E3": None, "E4": None, "E5": None, "E6": None,
}


def records(df):
    out = []
    for rec in df.to_dict("records"):
        row = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
                row[k] = None
            elif isinstance(v, float):
                row[k] = int(v) if v == int(v) and abs(v) < 1e9 else round(v, 6)
            elif hasattr(v, "item"):
                row[k] = v.item()
            else:
                row[k] = v
        out.append(row)
    return out


def build_data():
    data = {}
    for tid, lim in EMBED.items():
        data[tid] = {}
        for s in SUBS:
            df = R[tid][s]
            if lim == "rank5":
                df = df[df["소주제내순위"] <= 5]
            elif lim:
                df = df.head(lim)
            data[tid][s] = records(df)
    data["_meta"] = {
        "chapters": C.CHAPTERS, "short": {"제1장 선사 문화": "선사", "제2장 한국 고대사": "고대", "제3장 고려시대사": "고려",
                                           "제4장 조선시대사": "조선", "제5장 한국 근대사": "근대", "제6장 일제 강점기": "일제",
                                           "제7장 한국 현대사": "현대", "제8장 복합 문항": "복합", "특별 부록": "부록"},
        "nAll": len(C.Q), "nConf": len(C.SUBSETS["확인"]),
        "confEarlyPct": round(100 * sum(1 for x in C.SUBSETS["확인"] if C.chapter_of(x) in C.CHAPTERS[:4]) / len(C.SUBSETS["확인"]), 1), "rounds": C.ROUNDS, "next": C.NEXT,
    }
    return data


HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html"), encoding="utf-8").read()


def main():
    data = build_data()
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML.replace("/*__DATA__*/null", js)
    with open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("dashboard.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
