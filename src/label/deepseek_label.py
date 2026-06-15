import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DB_PATH = "banan_txt/output_json/labeled.db"
INPUT_FILE = "banan_txt/output_json/june7_laws_input.json"
OUTPUT_FILE = "banan_txt/output_json/june7_labeled_v2.jsonl"
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 128
BALANCE_EVERY_CALLS = 1000
BALANCE_EVERY_SECONDS = 1800

SYSTEM_PROMPT = """

Bạn là chuyên viên trích xuất trích dẫn pháp luật Việt Nam. Bạn nhận đầu vào là một câu văn bản pháp lý đã được đánh số thứ tự từng từ, định dạng "số#từ". Nhiệm vụ: liệt kê tất cả các cặp (danh sách số Điều, khoảng chỉ số của tên luật) mà câu đó đề cập. Mỗi luật khác nhau xuất ra một dòng riêng, định dạng:
ART: số_điều1,số_điều2,... | LAW: chỉ_số_đầu-chỉ_số_cuối

QUAN TRỌNG:
- Một câu có thể chứa nhiều luật (Bộ luật, Luật, Nghị quyết, Nghị định, v.v.). Với mỗi luật, hãy xác định khoảng chỉ số liên tục bao phủ toàn bộ tên luật đó (ví dụ "Bộ luật tố tụng dân sự" từ token 5 đến 10).
- Nếu luật được nhắc đến nhưng không có Điều nào, để trống ART: | LAW: ...
- Nếu nhiều Điều thuộc cùng một luật, ghi tất cả, phân cách bằng dấu phẩy. Thứ tự các Điều giữ nguyên như trong câu.
- Tên luật có thể bao gồm số (ví dụ "Nghị quyết số 326/2016/UBTVQH14", "Luật số 14/2017/QH14"). Phải bao gồm toàn bộ các token đó trong khoảng LAW.
- Khoảng LAW phải chính xác (bao đóng) và nằm trong độ dài câu. Nếu tên luật bị ngắt quãng bởi dấu câu hoặc từ khác, hãy gộp đúng liên tục từ đầu đến cuối tên.
- Đối với các trích dẫn dài, phức tạp (nhiều luật, nhiều điều, kèm năm, số hiệu), hãy đọc cẩn thận toàn bộ câu rồi lần lượt xuất từng dòng. Không bỏ sót luật nào.

CÁC VÍ DỤ (áp dụng cho mọi trường hợp, kể cả đầu vào dài):

Ví dụ 1 (một luật, một điều):
Input: 1#khoản 2#1 3#Điều 4#148 5#của 6#Bộ 7#luật 8#tố 9#tụng 10#dân 11#sự
Output: ART: 148 | LAW: 6-11

Ví dụ 2 (một luật, nhiều điều rải rác):
Input: 1#các 2#Điều 3#33, 4#38, 5#59 6#của 7#Luật 8#Hôn 9#nhân 10#và 11#gia 12#đình
Output: ART: 33,38,59 | LAW: 7-12

Ví dụ 3 (nhiều luật, mỗi luật có điều riêng):
Input: 1#các 2#Điều 3#33, 4#38 5#của 6#Luật 7#Hôn 8#nhân; 9#các 10#Điều 11#463, 12#466 13#của 14#Bộ 15#luật 16#Dân 17#sự
Output:
ART: 33,38 | LAW: 6-8
ART: 463,466 | LAW: 14-17

Ví dụ 4 (luật có số hiệu, không có điều):
Input: 1#Nghị 2#quyết 3#số 4#326/2016/UBTVQH14 5#ngày 6#30/12/2016 7#của 8#Ủy 9#ban 10#thường 11#vụ 12#Quốc 13#hội
Output: ART: | LAW: 1-13

Ví dụ 5 (trích dẫn hỗn hợp dài, giống đầu vào thực tế):
Input: 1#Khoản 2#9 3#Điều 4#26, 5#Điểm 6#a 7#Khoản 8#1 9#Điều 10#35, 11#Điểm 12#a, 13#c 14#Khoản 15#1 16#Điều 17#39, 18#Khoản 19#4 20#Điều 21#91, 22#Điều 23#94, 24#Điều 25#147, 26#Điều 27#157, 28#Điều 29#165, 30#Điều 31#166, 32#Điều 33#227, 34#Khoản 35#1 36#Điều 37#228, 38#Điều 39#271, 40#Điều 41#273 42#Bộ 43#luật 44#tố 45#tụng 46#dân 47#sự; 48#Điều 49#166 50#Bộ 51#luật 52#dân 53#sự 54#năm 55#2015; 56#Điều 57#4, 58#Điều 59#11, 60#Điều 61#26, 62#Điều 63#129, 64#Điều 65#131, 66#Điều 67#134, 68#Điều 69#138, 70#Điều 71#150, 72#Điều 73#235 74#và 75#Điều 76#236 77#Luật 78#Đất 79#đai 80#năm 81#2024
Output:
ART: 26,35,39,91,94,147,157,165,166,227,228,271,273 | LAW: 42-47
ART: 166 | LAW: 50-55
ART: 4,11,26,129,131,134,138,150,235,236 | LAW: 77-81

Ví dụ 6 (cực dài, nhiều luật xen kẽ, có năm và số hiệu):
Input: 1#Căn 2#cứ 3#khoản 4#2 5#Điều 6#26 7#Luật 8#Tổ 9#chức 10#Tòa 11#án 12#năm 13#2024; 14#khoản 15#3 16#Điều 17#12 18#Luật 19#Tố 20#tụng 21#hành 22#chính 23#năm 24#2015 25#(sửa 26#đổi 27#bổ 28#sung 29#năm 30#2019); 31#Điều 32#31 33#Nghị 34#định 35#số 36#01/2020/NĐ-CP 37#ngày 38#10/01/2020 39#của 40#Chính 41#phủ; 42#các 43#Điều 44#18, 45#19, 46#20 47#Bộ 48#luật 49#Dân 50#sự; 51#Điều 52#45 53#và 54#Điều 55#46 56#Bộ 57#luật 58#Tố 59#tụng 60#dân 61#sự; 62#khoản 63#1 64#Điều 65#7 66#Nghị 67#quyết 68#số 69#02/2023/NQ-HĐTP 70#ngày 71#15/02/2023 72#của 73#Hội 74#đồng 75#Thẩm 76#phán 77#Tòa 78#án 79#nhân 80#dân 81#tối 82#cao; 83#Điểm 84#c 85#khoản 86#2 87#Điều 88#21 89#Luật 90#Bảo 91#vệ 92#môi 93#trường 94#năm 95#2020.
Output:
ART: 26 | LAW: 7-13
ART: 12 | LAW: 18-30
ART: 31 | LAW: 33-41
ART: 18,19,20 | LAW: 47-50
ART: 45,46 | LAW: 56-61
ART: 7 | LAW: 66-82
ART: 21 | LAW: 89-95

Hãy chỉ xuất các dòng kết quả, không thêm bất kỳ văn bản nào khác. Đầu ra phải tuân thủ đúng định dạng ART: ... | LAW: ... mỗi dòng một cặp.

"""

client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


def fetch_balance_text():
    request = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8").strip()


async def print_balance():
    try:
        balance_text = await asyncio.to_thread(fetch_balance_text)
        print(f"[balance] {balance_text}")
    except Exception as exc:
        print(f"[balance] ERROR: {exc}")

def parse_args():
    parser = argparse.ArgumentParser(description="Label legal citations with optional resume support.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the SQLite database")
    return parser.parse_args()

def tokenize_and_index(text):
    tokens = text.split()
    indexed = " ".join(str(i+1) + "#" + tok for i, tok in enumerate(tokens))
    return indexed, tokens

def parse_output(raw_output, tokens):
    results = []
    for line in raw_output.strip().splitlines():
        if not line.strip():
            continue
        match = re.match(r'^ART:\s*(.*?)\s*\|\s*LAW:\s*(\d+)-(\d+)\s*$', line)
        if not match:
            continue
        art_part = match.group(1).strip()
        start_idx = int(match.group(2))
        end_idx = int(match.group(3))

        if art_part:
            articles = [a.strip() for a in art_part.split(',') if a.strip().isdigit()]
        else:
            articles = []

        if 1 <= start_idx <= end_idx <= len(tokens):
            law_name = ' '.join(tokens[start_idx-1:end_idx])
        else:
            law_name = "[INVALID_RANGE]"

        results.append((articles, law_name))
    return results

async def label_one(sem, entry_index, law_index, text, tokens, original_text):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=2048,
                temperature=0.3,
                stream=False
            )
            raw = resp.choices[0].message.content.strip()
            usage = resp.usage
            return entry_index, law_index, raw, usage.dict() if usage else None, tokens, original_text
        except Exception as e:
            return entry_index, law_index, f"ERROR: {str(e)}", None, tokens, original_text

async def main(resume=False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labeled_citations (
            file_name   TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            law_index   INTEGER NOT NULL,
            input_text  TEXT NOT NULL,
            output_text TEXT NOT NULL,
            usage_json  TEXT,
            citations_json TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (file_name, entry_index, law_index)
        )
    """)
    conn.commit()

    try:
        conn.execute("ALTER TABLE labeled_citations ADD COLUMN citations_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    completed_keys = set()
    if resume:
        cursor = conn.execute("SELECT file_name, entry_index, law_index FROM labeled_citations")
        completed_keys = set((row[0], row[1], row[2]) for row in cursor.fetchall())

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    
    for i, entry in enumerate(items):
        for j, law_text in enumerate(entry.get("laws_cited", [])):
            if resume and (entry.get("file_name", ""), i, j) in completed_keys:
                continue
            
            indexed_text, tokens = tokenize_and_index(law_text)
            tasks.append(label_one(sem, i, j, indexed_text, tokens, law_text))

    if not tasks:
        conn.close()
        return

    completed = 0
    last_balance_check = time.monotonic()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            entry_idx, law_idx, raw, usage, tokens, original_text = await coro
            completed += 1

            if completed % BALANCE_EVERY_CALLS == 0:
                await print_balance()
                last_balance_check = time.monotonic()
            elif time.monotonic() - last_balance_check >= BALANCE_EVERY_SECONDS:
                await print_balance()
                last_balance_check = time.monotonic()

            citations = parse_output(raw, tokens)
            citation_list = [{"articles": arts, "law": law} for arts, law in citations]

            record = {
                "file_name": items[entry_idx].get("file_name", ""),
                "entry_index": entry_idx,
                "law_index": law_idx,
                "input": original_text,
                "output_raw": raw,
                "citations": citation_list,
                "usage": usage
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            usage_json = json.dumps(usage) if usage else None
            citations_json = json.dumps(citation_list, ensure_ascii=False)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO labeled_citations "
                    "(file_name, entry_index, law_index, input_text, output_text, usage_json, citations_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        items[entry_idx].get("file_name", ""),
                        entry_idx,
                        law_idx,
                        original_text,
                        raw,
                        usage_json,
                        citations_json
                    )
                )
                conn.commit()
            except Exception:
                pass

    conn.close()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(resume=args.resume))