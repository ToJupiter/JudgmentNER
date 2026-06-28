import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MARKER_ORDER = ["preamble", "noi_dung_vu_an", "dai_dien_vks", "nhan_dinh", "quyet_dinh"]
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 512
BALANCE_EVERY_CALLS = 1000
BALANCE_EVERY_SECONDS = 1800

SYSTEM_PROMPT = """Bạn là bộ gán nhãn marker cấu trúc cho bản án Việt Nam. Đầu vào gồm một dòng ứng viên (CAND), các dòng trước (P1, P2), và dòng sau (N1, N2). Nhiệm vụ là xác định dòng ứng viên có phải dòng mở đầu một phần lớn của bản án hay không.

Các loại marker thật:
- preamble: NHÂN DANH hoặc dòng tiêu đề quốc hiệu mở đầu bản án
- noi_dung_vu_an: NỘI DUNG VỤ ÁN, NỘI DUNG BẢN ÁN, THEO CÁC TÀI LIỆU, QUÁ TRÌNH ĐIỀU TRA nếu dùng để mở phần nội dung
- dai_dien_vks: ĐẠI DIỆN VIỆN KIỂM SÁT, ĐẠI DIỆN VKS, Kiểm sát viên phát biểu/đề nghị nếu dùng để mở phần quan điểm VKS
- nhan_dinh: NHẬN ĐỊNH CỦA TÒA ÁN, NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ, HĐXX nhận định nếu dùng để mở phần nhận định
- quyet_dinh: QUYẾT ĐỊNH, VÌ CÁC LẼ TRÊN, TUYÊN XỬ nếu dùng để mở phần quyết định
5 loại marker này được xếp theo đúng thứ tự chúng xuất hiện trong bản án, do đó nên nếu bạn thấy dòng trên thuộc về loại trước thì chúng ta có thể đưa ra kết luận.

Marker giả là khi từ khóa nằm trong câu văn thường, trích dẫn lại bản án khác, tên văn bản, lời trình bày, hoặc không tạo boundary mới.

Định dạng đầu ra:
Chỉ trả về DUY NHẤT một chuỗi gồm đúng 3 chữ số viết liền, KHÔNG giải thích gì thêm, KHÔNG thêm bất kỳ văn bản nào khác. Trong đó:
- Chữ số 1 (Label): 0 (không phải marker thật), 1 (là marker thật)
- Chữ số 2 (Marker type): 0 (none), 1 (preamble), 2 (noi_dung_vu_an), 3 (dai_dien_vks), 4 (nhan_dinh), 5 (quyet_dinh)
- Chữ số 3 (Confidence): 1 (low/thấp), 2 (medium/trung bình), 3 (high/cao)

Dưới đây là 7 ví dụ minh họa:

Ví dụ 1 (Preamble thật):
REGEX_TYPE: preamble
P1:
CAND: NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
N1: Độc lập - Tự do - Hạnh phúc
OUT: 113

Ví dụ 2 (Nội dung vụ án thật):
REGEX_TYPE: noi_dung_vu_an
P1: Tòa án nhân dân tỉnh A thụ lý vụ án số 12.
CAND: NỘI DUNG VỤ ÁN:
N1: Theo đơn khởi kiện ngày 10/10/2025, nguyên đơn trình bày...
OUT: 123

Ví dụ 3 (Quan điểm VKS thật):
REGEX_TYPE: dai_dien_vks
P1: Sau khi các bên tranh luận xong.
CAND: ĐẠI DIỆN VIỆN KIỂM SÁT PHÁT BIỂU Ý KIẾN:
N1: Kiểm sát viên trình bày quan điểm giải quyết vụ án...
OUT: 133

Ví dụ 4 (Nhận định thật):
REGEX_TYPE: nhan_dinh
P1: Đại diện Viện kiểm sát đề nghị xử phạt bị cáo từ 2 đến 3 năm tù.
CAND: NHẬN ĐỊNH CỦA TÒA ÁN:
N1: Trên cơ sở nội dung vụ án, căn cứ vào các tài liệu...
OUT: 143

Ví dụ 5 (Quyết định thật):
REGEX_TYPE: quyet_dinh
P1: Vì các lẽ trên, Hội đồng xét xử quyết định chấp nhận một phần yêu cầu khởi kiện.
CAND: QUYẾT ĐỊNH:
N1: Căn cứ vào các điều luật... tuyên xử bị cáo A...
OUT: 153

Ví dụ 6 (Giả/Âm tính - từ khóa nằm trong câu thường):
REGEX_TYPE: quyet_dinh
P1: Sau khi nghị án, Hội đồng xét xử nhận định.
CAND: Hội đồng xét xử quyết định hình phạt đối với bị cáo là phù hợp.
N1: Bị cáo phải chịu án phí theo quy định pháp luật.
OUT: 003

Ví dụ 7 (Giả/Âm tính - trích dẫn lại bản án khác):
REGEX_TYPE: nhan_dinh
P1: Nguyên đơn không đồng ý với bản án sơ thẩm.
CAND: Bản án sơ thẩm có phần nhận định chưa khách quan và chính xác.
N1: Do đó nguyên đơn kháng cáo toàn bộ bản án.
OUT: 003
"""

USER_TEMPLATE = """
FILE: {file_name}
LINE_INDEX: {line_index}
REGEX_TYPE: {expected_marker_type}
KIND: {candidate_kind}
POS: {position_ratio}
P2: {previous_2}
P1: {previous_1}
CAND: {candidate_line}
N1: {next_1}
N2: {next_2}
"""


def safe_text(x: Any) -> str:
    return x if isinstance(x, str) else ""


def parse_json_maybe(text: str) -> dict:
    text = safe_text(text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def confidence_rank(x: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(safe_text(x).lower(), 0)


def normalize_line(x: str) -> str:
    return re.sub(r"\s+", " ", safe_text(x)).strip()


def looks_title_like(line: str) -> bool:
    x = normalize_line(line)
    if len(x) <= 100:
        return True
    letters = [c for c in x if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.upper() == c)
    return upper / max(1, len(letters)) >= 0.55


def rule_label(candidate: dict) -> dict | None:
    line = normalize_line(candidate.get("candidate_line", ""))
    up = line.upper()
    expected = candidate.get("expected_marker_type", "none")
    strong = {
        "preamble": ["NHÂN DANH", "NHAN DANH", "NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"],
        "noi_dung_vu_an": ["NỘI DUNG VỤ ÁN", "NOI DUNG VU AN", "NỘI DUNG BẢN ÁN", "NOI DUNG BAN AN"],
        "dai_dien_vks": ["ĐẠI DIỆN VIỆN KIỂM SÁT", "DAI DIEN VIEN KIEM SAT", "ĐẠI DIỆN VKS"],
        "nhan_dinh": ["NHẬN ĐỊNH CỦA TÒA ÁN", "NHAN DINH CUA TOA AN", "NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ", "HĐXX NHẬN ĐỊNH"],
        "quyet_dinh": ["QUYẾT ĐỊNH", "QUYET DINH"]
    }
    if expected in strong and any(x in up for x in strong[expected]) and looks_title_like(line):
        out = dict(candidate)
        out.update({"label": 1, "marker_type": expected, "confidence": "high", "reason": "rule_strong_marker", "raw_response": "", "used_llm": False})
        return out
    if expected == "quyet_dinh" and ("VÌ CÁC LẼ TRÊN" in up or "VI CAC LE TREN" in up or re.match(r"^\s*TUY[ÊE]N\s+XỬ\s*:?.*$", up, flags=re.I)):
        out = dict(candidate)
        out.update({"label": 1, "marker_type": "quyet_dinh", "confidence": "medium", "reason": "rule_decision_transition", "raw_response": "", "used_llm": False})
        return out
    if candidate.get("candidate_kind") == "random_negative":
        out = dict(candidate)
        out.update({"label": 0, "marker_type": "none", "confidence": "high", "reason": "rule_random_negative", "raw_response": "", "used_llm": False})
        return out
    return None


def flatten_candidates(data: Any) -> list[dict]:
    if isinstance(data, dict):
        data = [data]
    out = []
    for doc in data:
        for c in doc.get("candidates", []):
            item = dict(c)
            item.setdefault("file_name", doc.get("file_name", ""))
            item.setdefault("path", doc.get("path", ""))
            out.append(item)
    return out


def fetch_balance_text() -> str:
    req = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        headers={"Accept": "application/json", "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        method="GET"
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8").strip()


async def print_balance():
    try:
        text = await asyncio.to_thread(fetch_balance_text)
        print(f"[balance] {text}")
    except Exception as exc:
        print(f"[balance] ERROR: {exc}")


async def label_one(client: AsyncOpenAI, sem: asyncio.Semaphore, candidate: dict, model_name: str, max_retries: int) -> dict:
    rule = rule_label(candidate)
    prompt = USER_TEMPLATE.format(**candidate)
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    extra_body={"thinking": {"type": "disabled"}},
                    temperature=0.0,
                    max_tokens=10,
                    stream=False
                )
            raw = safe_text(resp.choices[0].message.content).strip()
            
            # Parse 3-digit output, e.g., 113
            match = re.search(r"([0-1])([0-5])([1-3])", raw)
            if match:
                label = int(match.group(1))
                marker_idx = match.group(2)
                conf_idx = match.group(3)
                
                marker_map = {
                    "0": "none",
                    "1": "preamble",
                    "2": "noi_dung_vu_an",
                    "3": "dai_dien_vks",
                    "4": "nhan_dinh",
                    "5": "quyet_dinh"
                }
                conf_map = {
                    "1": "low",
                    "2": "medium",
                    "3": "high"
                }
                
                marker_type = marker_map.get(marker_idx, "none")
                confidence = conf_map.get(conf_idx, "low")
            else:
                label = 0
                marker_type = "none"
                confidence = "low"
            
            reason = ""
            if label != 1:
                marker_type = "none"
            if marker_type not in MARKER_ORDER and marker_type != "none":
                marker_type = candidate.get("expected_marker_type", "none")
            if confidence_rank(confidence) == 0:
                confidence = "low"
                
            out = dict(candidate)
            if rule is not None:
                out.update({
                    "rule_label": rule.get("label"),
                    "rule_marker_type": rule.get("marker_type"),
                    "rule_confidence": rule.get("confidence"),
                    "rule_reason": rule.get("reason")
                })
            else:
                out.update({
                    "rule_label": None,
                    "rule_marker_type": None,
                    "rule_confidence": None,
                    "rule_reason": None
                })
                
            out.update({
                "label": label,
                "marker_type": marker_type,
                "confidence": confidence,
                "reason": reason,
                "raw_response": raw,
                "used_llm": True,
                "usage": resp.usage.dict() if resp.usage else None
            })
            return out
        except Exception as exc:
            if attempt + 1 >= max_retries:
                out = dict(candidate)
                if rule is not None:
                    out.update({
                        "rule_label": rule.get("label"),
                        "rule_marker_type": rule.get("marker_type"),
                        "rule_confidence": rule.get("confidence"),
                        "rule_reason": rule.get("reason")
                    })
                else:
                    out.update({
                        "rule_label": None,
                        "rule_marker_type": None,
                        "rule_confidence": None,
                        "rule_reason": None
                    })
                out.update({"label": 0, "marker_type": "none", "confidence": "low", "reason": f"llm_error:{exc}", "raw_response": "", "used_llm": True, "usage": None})
                return out
            await asyncio.sleep(1.5 * (attempt + 1))
            
    out = dict(candidate)
    if rule is not None:
        out.update({
            "rule_label": rule.get("label"),
            "rule_marker_type": rule.get("marker_type"),
            "rule_confidence": rule.get("confidence"),
            "rule_reason": rule.get("reason")
        })
    else:
        out.update({
            "rule_label": None,
            "rule_marker_type": None,
            "rule_confidence": None,
            "rule_reason": None
        })
    out.update({"label": 0, "marker_type": "none", "confidence": "low", "reason": "unreachable", "raw_response": "", "used_llm": True, "usage": None})
    return out


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labeled_chunk_markers (
            file_name TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            line_index INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (file_name, candidate_id)
        )
    """)
    conn.commit()
    return conn


def completed_keys(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = conn.execute("SELECT file_name, candidate_id FROM labeled_chunk_markers").fetchall()
    return {(r[0], r[1]) for r in rows}


def read_existing_outputs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT output_json FROM labeled_chunk_markers ORDER BY file_name, line_index, candidate_id").fetchall()
    out = []
    for (x,) in rows:
        try:
            out.append(json.loads(x))
        except Exception:
            pass
    return out


def write_record(conn: sqlite3.Connection, row: dict):
    conn.execute(
        "INSERT OR REPLACE INTO labeled_chunk_markers (file_name, candidate_id, line_index, input_json, output_json) VALUES (?, ?, ?, ?, ?)",
        (row.get("file_name", ""), row.get("candidate_id", ""), int(row.get("line_index", -1)), json.dumps({k: row.get(k) for k in ["file_name", "candidate_id", "line_index", "candidate_line", "previous_1", "next_1"]}, ensure_ascii=False), json.dumps(row, ensure_ascii=False))
    )
    conn.commit()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=Path, required=True)
    parser.add_argument("-o", "--output-file", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("chunk_marker_labels.db"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.input_file.read_text(encoding="utf-8"))
    candidates = flatten_candidates(data)
    if args.limit > 0:
        candidates = candidates[:args.limit]

    conn = open_db(args.db_path)
    done = completed_keys(conn) if args.resume else set()
    pending = [c for c in candidates if (c.get("file_name", ""), c.get("candidate_id", "")) not in done]
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        existing = read_existing_outputs(conn)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in existing:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    completed = 0
    last_balance = time.monotonic()
    with open(args.output_file, "a", encoding="utf-8") as out:
        tasks = [label_one(client, sem, c, args.model_name, args.max_retries) for c in pending]
        for coro in asyncio.as_completed(tasks):
            row = await coro
            completed += 1
            write_record(conn, row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if completed % BALANCE_EVERY_CALLS == 0:
                await print_balance()
                last_balance = time.monotonic()
            elif time.monotonic() - last_balance >= BALANCE_EVERY_SECONDS:
                await print_balance()
                last_balance = time.monotonic()
            if completed % 200 == 0:
                print(json.dumps({"done": completed, "pending_total": len(pending)}, ensure_ascii=False))

    conn.close()
    print(json.dumps({"labeled": completed, "output": str(args.output_file)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
