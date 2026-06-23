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

DB_PATH = "banan_txt/output_json/labeled_2.db"
INPUT_FILE = "banan_txt/output_json/june23_laws.json"
OUTPUT_FILE = "banan_txt/output_json/june23_laws_labeled.jsonl"
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 128
BALANCE_EVERY_CALLS = 1000
BALANCE_EVERY_SECONDS = 1800

SYSTEM_PROMPT = """
Bạn là chuyên viên trích xuất trích dẫn pháp luật Việt Nam. Bạn nhận đầu vào là một câu văn bản pháp lý đã được đánh số thứ tự từng từ, định dạng "số#từ". Nhiệm vụ của bạn là liệt kê tất cả các cặp (danh sách số Điều, khoảng chỉ số của tên luật) mà câu đó đề cập. Mỗi luật khác nhau – hoặc mỗi lần xuất hiện riêng biệt của cùng một luật với tập Điều riêng – được xuất ra trên một dòng riêng, định dạng:
ART: số_điều1,số_điều2,... | LAW: chỉ_số_đầu-chỉ_số_cuối

QUY TẮC CHI TIẾT:
- Luật có thể xuất hiện dưới nhiều hình thức: "Bộ luật …", "Luật …", "Nghị quyết số …", "Nghị định số …", "Bộ luật … năm …", "Luật … sửa đổi bổ sung năm …", … Hãy giữ nguyên toàn bộ tên luật, bao gồm số hiệu, năm và phần bổ sung, nếu có. Khoảng LAW phải bao phủ toàn bộ các token liên tục tạo thành tên luật, từ token đầu tiên cho đến token cuối cùng thuộc tên đó (loại trừ dấu câu thừa, chú thích ngoài lề như "(10%/năm)" sau tên).
- Với mỗi luật, liệt kê tất cả số Điều (số nguyên) được nhắc đến cùng luật đó. Các số Điều được liệt kê theo đúng thứ tự xuất hiện trong câu, phân cách bằng dấu phẩy. Nếu luật được nhắc đến nhưng không có Điều nào đi kèm, hãy xuất dòng có ART để trống: ART: | LAW: ...
- Một câu có thể chứa nhiều luật. Nếu cùng một luật xuất hiện nhiều lần với các tập Điều khác nhau (tách biệt bởi nội dung khác), mỗi lần xuất hiện được coi là một dòng riêng, với khoảng LAW tương ứng với đúng vị trí tên luật trong lần xuất hiện đó.
- Chỉ số token là số nguyên dương 1‑based. Khoảng chỉ số là bao đóng (gồm cả hai đầu). Khoảng LAW phải chính xác, không được thừa hay thiếu token nào thuộc tên luật.
- Đối với các đầu vào dài, phức tạp, hãy đọc toàn bộ câu một cách cẩn thận, lần lượt trích xuất từng luật, đảm bảo không bỏ sót bất kỳ luật nào.
- Chỉ xuất ra các dòng kết quả, không thêm bất kỳ văn bản nào khác (không giải thích, không tiêu đề).

DƯỚI ĐÂY LÀ CÁC VÍ DỤ MINH HỌA (áp dụng cho mọi độ dài đầu vào):

Ví dụ 1 – một điều, một luật:
Input: 1#khoản 2#4 3#Điều 4#174 5#Bộ 6#luật 7#Hình 8#sự.
Output: ART: 174 | LAW: 5-8

Ví dụ 2 – nhiều điều, một luật:
Input: 1#các 2#Điều 3#271, 4#Điều 5#272 6#và 7#Điều 8#273 9#Bộ 10#luật 11#tố 12#tụng 13#Dân 14#sự.
Output: ART: 271,272,273 | LAW: 9-14

Ví dụ 3 – luật không có điều:
Input: 1#Nghị 2#quyết 3#số 4#326/2016/UBTVQH14 5#ngày 6#30/12/2016 7#của 8#Ủy 9#ban 10#thường 11#vụ 12#Quốc 13#hội
Output: ART: | LAW: 1-13

Ví dụ 4 – nhiều luật xen kẽ, có luật không điều:
Input: 1#các 2#Điều 3#33, 4#38, 5#59 6#của 7#Luật 8#Hôn 9#nhân 10#và 11#gia 12#đình; 13#các 14#Điều 15#463, 16#466, 17#357, 18#468 19#của 20#Bộ 21#luật 22#Dân 23#sự; 24#Nghị 25#quyết 26#số 27#326/2016/UBTVQH14
Output:
ART: 33,38,59 | LAW: 7-12
ART: 463,466,357,468 | LAW: 20-23
ART: | LAW: 24-27

Ví dụ 5 – tên luật có năm, chú thích ngoài được loại bỏ:
Input: 1#khoản 2#2 3#Điều 4#468 5#Bộ 6#luật 7#Dân 8#sự 9#(10%/năm).
Output: ART: 468 | LAW: 5-8

Ví dụ 6 – đoạn dài phức tạp (nhiều luật, nhiều điều, có năm và số hiệu):
Input: 1#Khoản 2#9 3#Điều 4#26, 5#Điểm 6#a 7#Khoản 8#1 9#Điều 10#35, 11#Điểm 12#a, 13#c 14#Khoản 15#1 16#Điều 17#39, 18#Khoản 19#4 20#Điều 21#91, 22#Điều 23#94, 24#Điều 25#147, 26#Điều 27#157, 28#Điều 29#165, 30#Điều 31#166, 32#Điều 33#227, 34#Khoản 35#1 36#Điều 37#228, 38#Điều 39#271, 40#Điều 41#273 42#Bộ 43#luật 44#tố 45#tụng 46#dân 47#sự; 48#Điều 49#166 50#Bộ 51#luật 52#dân 53#sự 54#năm 55#2015; 56#Điều 57#4, 58#Điều 59#11, 60#Điều 61#26, 62#Điều 63#129, 64#Điều 65#131, 66#Điều 67#134, 68#Điều 69#138, 70#Điều 71#150, 72#Điều 73#235 74#và 75#Điều 76#236 77#Luật 78#Đất 79#đai 80#năm 81#2024
Output:
ART: 26,35,39,91,94,147,157,165,166,227,228,271,273 | LAW: 42-47
ART: 166 | LAW: 50-55
ART: 4,11,26,129,131,134,138,150,235,236 | LAW: 77-81

Ví dụ 7 – đầu vào rất dài, nhiều luật có sửa đổi bổ sung:
Input: 1#Căn 2#cứ 3#khoản 4#2 5#Điều 6#26 7#Luật 8#Tổ 9#chức 10#Tòa 11#án 12#năm 13#2024; 14#khoản 15#3 16#Điều 17#12 18#Luật 19#Tố 20#tụng 21#hành 22#chính 23#năm 24#2015 25#(sửa 26#đổi 27#bổ 28#sung 29#năm 30#2019); 31#Điều 32#31 33#Nghị 34#định 35#số 36#01/2020/NĐ-CP 37#ngày 38#10/01/2020 39#của 40#Chính 41#phủ; 42#các 43#Điều 44#18, 45#19, 46#20 47#Bộ 48#luật 49#Dân 50#sự; 51#Điều 52#45 53#và 54#Điều 55#46 56#Bộ 57#luật 58#Tố 59#tụng 60#dân 61#sự; 62#khoản 63#1 64#Điều 65#7 66#Nghị 67#quyết 68#số 69#02/2023/NQ-HĐTP 70#ngày 71#15/02/2023 72#của 73#Hội 74#đồng 75#Thẩm 76#phán 77#Tòa 78#án 79#nhân 80#dân 81#tối 82#cao; 83#Điểm 84#c 85#khoản 86#2 87#Điều 88#21 89#Luật 90#Bảo 91#vệ 92#môi 93#trường 94#năm 95#2020.
Output:
ART: 26 | LAW: 7-13
ART: 12 | LAW: 18-30
ART: 31 | LAW: 33-41
ART: 18,19,20 | LAW: 47-50
ART: 45,46 | LAW: 56-61
ART: 7 | LAW: 66-82
ART: 21 | LAW: 89-95

Ví dụ 8 – không có trích dẫn luật nào (với các ví dụ không có trích dẫn luật, chỉ để 1 output duy nhất là "NL", có nghĩa NO LAW):
Input: 1#các 2#điểm 3#(đỉnh 4#thửa): 5#A-B-C-D 6#thể 7#hiện 8#tại 9#Mảnh 10#trích 11#đo.
Output: NL

Hãy chỉ xuất đúng các dòng kết quả, không thêm bất kỳ văn bản nào khác.
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