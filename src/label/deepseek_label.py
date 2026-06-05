import argparse
import asyncio
import json
import os
import re
import sqlite3
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DB_PATH = "banan_txt/output_json/labeled.db"
INPUT_FILE = "banan_txt/output_json/june6_laws_filtered.json"
OUTPUT_FILE = "banan_txt/output_json/june6_labeled.jsonl"
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 128

SYSTEM_PROMPT = """Bạn là chuyên viên trích xuất trích dẫn pháp luật Việt Nam. Nhiệm vụ của bạn là đọc một câu văn bản pháp lý đã được đánh chỉ số từ và liệt kê tất cả các cặp (danh sách số Điều, khoảng chỉ số của tên luật) mà câu đó đề cập.

Mỗi từ trong câu được đánh số thứ tự bắt đầu từ 1, định dạng "số#từ". Ví dụ: "1#khoản 2#4 3#Điều 4#174 5#Bộ 6#luật 7#Hình 8#sự."

Luật có thể xuất hiện ở nhiều dạng: "Bộ luật …", "Luật …", "Nghị quyết số …", "Bộ luật … năm …", "Luật … sửa đổi bổ sung năm …", "Nghị định số...". Nếu tên luật có kèm năm và phần bổ sung thì giữ nguyên toàn bộ phần tên đó.

Một câu có thể chứa nhiều luật khác nhau; mỗi cặp ghi trên một dòng riêng, định dạng:
ART: số_điều1,số_điều2,... | LAW: chỉ_số_đầu-chỉ_số_cuối

Nếu một luật được nhắc đến nhưng không có Điều nào kèm theo, bạn vẫn xuất dòng đó với phần ART để trống: ART: | LAW: chỉ_số_đầu-chỉ_số_cuối

Chỉ số là số nguyên dương (1-based) tương ứng với vị trí từ trong câu nhập vào. Khoảng chỉ số là bao đóng (gồm cả hai đầu). Chỉ xuất các dòng kết quả, không thêm bất cứ văn bản nào khác.

Dưới đây là các ví dụ minh họa.

Ví dụ 1:
Input: 1#khoản 2#4 3#Điều 4#174 5#Bộ 6#luật 7#Hình 8#sự.
Output: ART: 174 | LAW: 5-8

Ví dụ 2:
Input: 1#các 2#Điều 3#271, 4#Điều 5#272 6#và 7#Điều 8#273 9#Bộ 10#luật 11#tố 12#tụng 13#Dân 14#sự, 15#được 16#chấp 17#nhận 18#xem 19#xét 20#theo 21#thủ 22#tục 23#phúc 24#thẩm.
Output: ART: 271,272,273 | LAW: 9-14

Ví dụ 3:
Input: 1#khoản 2#1 3#Điều 4#308 5#Bộ 6#luật 7#tố 8#tụng 9#Dân 10#sự: 11#Đề 12#nghị 13#Hội 14#đồng 15#xét 16#xử 17#phúc 18#thẩm 19#Tòa 20#án 21#nhân 22#dân 23#tỉnh 24#Điện 25#Biên 26#giữ 27#nguyên 28#Bản 29#án 30#dân 31#sự 32#sơ 33#thẩm 34#số 35#04/2025/DS-ST 36#ngày 37#28/9/2025 38#của 39#Tòa 40#án 41#nhân 42#dân 43#khu 44#vực 45#3, 46#tỉnh 47#Điện 48#Biên.
Output: ART: 308 | LAW: 5-10

Ví dụ 4:
Input: 1#các 2#Điều 3#33, 4#38, 5#59 6#của 7#Luật 8#Hôn 9#nhân 10#và 11#gia 12#đình; 13#các 14#Điều 15#463, 16#466, 17#357, 18#468 19#của 20#Bộ 21#luật 22#Dân 23#sự; 24#Nghị 25#quyết 26#số 27#326/2016/UBTVQH14
Output:
ART: 33,38,59 | LAW: 7-12
ART: 463,466,357,468 | LAW: 20-23
ART: | LAW: 24-27

Ví dụ 5:
Input: 1#khoản 2#2 3#Điều 4#468 5#Bộ 6#luật 7#Dân 8#sự 9#(10%/năm).
Output: ART: 468 | LAW: 5-8

Ví dụ 6:
Input: 1#Điều 2#26, 3#31, 4#235, 5#236 6#Luật 7#đất 8#đai 9#2024; 10#Điều 11#115, 12#Điều 13#158, 14#Điều 15#159, 16#Điều 17#165, 18#Điều 19#166, 20#Điều 21#175, 22#Điều 23#176 24#Bộ 25#luật 26#dân 27#sự; 28#Căn 29#cứ 30#khoản 31#2 32#Điều 33#26 34#Nghị 35#quyết 36#số 37#326/2016/UBTVQH 38#ngày 39#30/12/2016 40#của 41#Ủy 42#ban 43#thường 44#vu 45#Quốc 46#hội 47#quy 48#định 49#về 50#án 51#phí, 52#lệ 53#phí 54#Tòa 55#án.
Output:
ART: 26,31,235,236 | LAW: 6-9
ART: 115,158,159,165,166,175,176 | LAW: 24-27
ART: 26 | LAW: 34-55

Ví dụ 7:
Input: 1#khoản 2#1 3#Điều 4#308; 5#Điều 6#148 7#của 8#Bộ 9#luật 10#Tố 11#tụng 12#dân 13#sự, 14#Nghị 15#quyết 16#số 17#326/2016/UBTVQH14 18#ngày 19#30/12/2016 20#quy 21#định 22#về 23#mức 24#thu, 25#miễn, 26#giảm, 27#thu, 28#nộp, 29#quản 30#lý 31#và 32#sử 33#dụng 34#phí, 35#án 36#phí 37#Tòa 38#án 39#của 40#Ủy 41#ban 42#thường 43#vu 44#Quốc 45#hội.
Output:
ART: 308,148 | LAW: 8-13
ART: | LAW: 14-45

Ví dụ 8:
Input: 1#khoản 2#2 3#Điều 4#3 5#Luật 6#số 7#14/2017/QH14 8#của 9#Quốc 10#Hội 11#về 12#quản 13#lý, 14#sử 15#dụng 16#vũ 17#khí, 18#vật 19#liệu 20#nổ 21#và 22#công 23#cụ 24#hỗ 25#trợ 26#thì 27#bật 28#lửa 29#hình 30#lựu 31#đạn 32#này 33#không 34#thuộc 35#vũ 36#khí 37#quân 38#dụng.
Output: ART: 3 | LAW: 5-25

Ví dụ 9:
Input: 1#các 2#điểm 3#(đỉnh 4#thửa): 5#A-B-C-E-F-G-H-K-L-M 6#thể 7#hiện 8#tại 9#Mảnh 10#trích 11#đo 12#địa 13#chính 14#số 15#36- 16#2025 17#do 18#Công 19#ty 20#cổ 21#phần 22#đo 23#đạc 24#N 25#thực 26#hiện 27#ngày 28#10 29#tháng 30#4 31#năm 32#2025 33#kèm 34#theo 35#Bản 36#án.
Output: (không xuất dòng nào)

Ví dụ 10:
Input: 1#Điểm 2#3: 3#1244848. 4#Điểm 5#4: 6#1244849.
Output: (không xuất dòng nào)

Ví dụ 11:
Input: 1#khoản 2#1 3#Điều 4#14 5#Nghị 6#định 7#số 8#91/2019/NĐ- 9#CP 10#ngày 11#19/11/2019 12#của 13#Chính 14#phủ 15#về 16#xử 17#phạt 18#vi 19#phạm 20#hành 21#chính 22#trong 23#lĩnh 24#vực 25#đất 26#đai.
Output: ART: 14 | LAW: 5-26"""

client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

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
    total = len(tasks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            entry_idx, law_idx, raw, usage, tokens, original_text = await coro
            completed += 1

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
            except Exception as e:
                pass

    conn.close()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(resume=args.resume))