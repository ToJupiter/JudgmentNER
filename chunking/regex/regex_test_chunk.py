import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

MARKER_ORDER = [
    "preamble",
    "noi_dung_vu_an",
    "dai_dien_vks",
    "nhan_dinh",
    "quyet_dinh"
]

MARKER_PATTERNS = {
    "preamble": [
        r"^\s*#{0,6}\s*(?:\d+[.)-]\s*)?NHÂN\s+DANH\b.*$",
        r"^\s*#{0,6}\s*(?:\d+[.)-]\s*)?NHAN\s+DANH\b.*$",
        r"NƯỚC\s+CỘNG\s+H[ÒO]A\s+X[ÃA]\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM",
        r"NUOC\s+CONG\s+HOA\s+XA\s+HOI\s+CHU\s+NGHIA\s+VIET\s+NAM"
    ],
    "noi_dung_vu_an": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NỘI\s+DUNG\s+VỤ\s+ÁN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NOI\s+DUNG\s+VU\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NỘI\s+DUNG\s+BẢN\s+ÁN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NOI\s+DUNG\s+BAN\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*THEO\s+C[ÁA]C\s+T[ÀA]I\s+LIỆU.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QU[ÁA]\s+TR[ÌI]NH\s+(?:ĐIỀU\s+TRA|GIẢI\s+QUYẾT).*$"
    ],
    "dai_dien_vks": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*ĐẠI\s+DIỆN\s+VIỆN\s+KIỂM\s+S[ÁA]T.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*DAI\s+DIEN\s+VIEN\s+KIEM\s+SAT.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*ĐẠI\s+DIỆN\s+VKS.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*KIỂM\s+S[ÁA]T\s+VI[ÊE]N.*(?:ĐỀ\s+NGHỊ|PH[ÁA]T\s+BIỂU).*$",
        r"VIỆN\s+KIỂM\s+S[ÁA]T.*(?:ĐỀ\s+NGHỊ|PH[ÁA]T\s+BIỂU|QUAN\s+ĐIỂM)"
    ],
    "nhan_dinh": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHẬN\s+ĐỊNH\s+CỦA\s+T[ÒO]A\s+[ÁA]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHAN\s+DINH\s+CUA\s+TOA\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHẬN\s+ĐỊNH\s+CỦA\s+HỘI\s+ĐỒNG\s+X[ÉE]T\s+XỬ\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*HỘI\s+ĐỒNG\s+X[ÉE]T\s+XỬ\s+NHẬN\s+ĐỊNH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*HĐXX\s+NHẬN\s+ĐỊNH\s*:?.*$"
    ],
    "quyet_dinh": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QUYẾT\s+ĐỊNH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QUYET\s+DINH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*V[ÌI]\s+C[ÁA]C\s+LẼ\s+TR[ÊE]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*V[ÌI]\s+C[ÁA]C\s+L[ẼE]\s+TR[ÊE]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*TUY[ÊE]N\s+XỬ\s*:?.*$"
    ]
}

HARD_NEGATIVE_PATTERNS = [
    r"\bnhân\s+danh\b",
    r"\bnội\s+dung\b",
    r"\bviện\s+kiểm\s+s[áa]t\b",
    r"\bkiểm\s+s[áa]t\s+vi[êe]n\b",
    r"\bnhận\s+định\b",
    r"\bquyết\s+định\b",
    r"\btuyên\s+xử\b",
    r"\bhđxx\b",
    r"\bhội\s+đồng\s+x[ée]t\s+xử\b"
]


def safe_text(x: Any) -> str:
    return x if isinstance(x, str) else ""


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", safe_text(line))
    line = line.replace("\ufeff", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_lines(text: str) -> list[str]:
    return [normalize_line(x) for x in safe_text(text).splitlines()]


def compact_nonempty_lines(lines: list[str]) -> list[str]:
    return [x for x in lines if x.strip()]


def compile_patterns(patterns: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {k: [re.compile(p, flags=re.I | re.U) for p in v] for k, v in patterns.items()}


def marker_strength(line: str, marker_type: str) -> float:
    x = normalize_line(line)
    y = x.upper()
    short = len(x) <= 100
    heading = x.startswith("#") or bool(re.match(r"^\s*(?:[IVX]+|\d+)[.)-]", x, flags=re.I))
    all_caps = sum(1 for c in x if c.isalpha() and c.upper() == c) >= max(3, int(sum(1 for c in x if c.isalpha()) * 0.6))
    score = 1.0
    if short:
        score += 1.0
    if heading:
        score += 0.7
    if all_caps:
        score += 0.7
    if marker_type == "preamble" and ("NHÂN DANH" in y or "NHAN DANH" in y):
        score += 1.0
    if marker_type == "quyet_dinh" and ("QUYẾT ĐỊNH" in y or "QUYET DINH" in y):
        score += 1.0
    return score


def make_candidate(file_name: str, lines: list[str], i: int, expected: str, kind: str, score: float) -> dict:
    n = len(lines)
    return {
        "file_name": file_name,
        "candidate_id": f"{file_name}:{i}:{expected}:{kind}",
        "line_index": i,
        "expected_marker_type": expected,
        "candidate_kind": kind,
        "candidate_line": lines[i],
        "previous_2": lines[i - 2] if i - 2 >= 0 else "",
        "previous_1": lines[i - 1] if i - 1 >= 0 else "",
        "next_1": lines[i + 1] if i + 1 < n else "",
        "next_2": lines[i + 2] if i + 2 < n else "",
        "position_ratio": i / max(1, n - 1),
        "regex_score": score
    }


def find_marker_candidates(file_name: str, lines: list[str], negative_per_file: int, seed: int) -> list[dict]:
    compiled = compile_patterns(MARKER_PATTERNS)
    hard_compiled = [re.compile(p, flags=re.I | re.U) for p in HARD_NEGATIVE_PATTERNS]
    candidates = []
    seen = set()

    for i, line in enumerate(lines):
        if not line:
            continue
        for marker_type, patterns in compiled.items():
            for pattern in patterns:
                if pattern.search(line):
                    key = (i, marker_type, "regex")
                    if key not in seen:
                        seen.add(key)
                        candidates.append(make_candidate(file_name, lines, i, marker_type, "regex", marker_strength(line, marker_type)))
                    break

    for i, line in enumerate(lines):
        if not line:
            continue
        if any(p.search(line) for p in hard_compiled):
            if not any(c["line_index"] == i for c in candidates):
                key = (i, "none", "hard_negative")
                if key not in seen:
                    seen.add(key)
                    candidates.append(make_candidate(file_name, lines, i, "none", "hard_negative", 0.2))

    rng = random.Random(seed + sum(ord(c) for c in file_name))
    possible = [i for i, line in enumerate(lines) if line and not any(c["line_index"] == i for c in candidates)]
    rng.shuffle(possible)
    for i in possible[:max(0, negative_per_file)]:
        candidates.append(make_candidate(file_name, lines, i, "none", "random_negative", 0.0))

    candidates.sort(key=lambda x: (x["line_index"], x["expected_marker_type"], x["candidate_kind"]))
    return candidates


def process_folder(folder: Path, output_file: Path, negative_per_file: int, seed: int) -> None:
    results = []
    total_candidates = 0
    for md_file in sorted(folder.glob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        lines = compact_nonempty_lines(split_lines(text))
        candidates = find_marker_candidates(md_file.name, lines, negative_per_file, seed)
        total_candidates += len(candidates)
        results.append({
            "file_name": md_file.name,
            "path": str(md_file),
            "line_count": len(lines),
            "candidates": candidates,
            "candidate_count": len(candidates)
        })
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(results), "candidates": total_candidates, "output": str(output_file)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder-name", type=Path, required=True)
    parser.add_argument("-o", "--output-file", type=Path, required=True)
    parser.add_argument("--negative-per-file", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_folder(args.folder_name, args.output_file, args.negative_per_file, args.seed)


if __name__ == "__main__":
    main()
