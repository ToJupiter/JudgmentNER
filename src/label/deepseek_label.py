import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

INPUT_FILE = "/media/rocminfo/565A28A25A2880BB/VBPL/CBBACrawl/banan_txt/output_json"
OUTPUT_FILE = "/media/rocminfo/565A28A25A2880BB/VBPL/CBBACrawl/banan_txt/output_json/june5_labeled.jsonl"
MODEL_NAME = "deepseek-v4-flash"
MAX_CONCURRENT = 128

SYSTEM_PROMPT = """
Bạn là chuyên viên trích xuất trích dẫn pháp luật Việt Nam. Nhiệm vụ của bạn là đọc một câu văn bản pháp lý và liệt kê tất cả các cặp (danh sách số Điều, tên luật) mà câu đó đề cập.

Luật có thể xuất hiện ở nhiều dạng: "Bộ luật …", "Luật …", "Nghị quyết số …", "Bộ luật … năm …", "Luật … sửa đổi bổ sung năm …", "Nghị định số...". Nếu tên luật có kèm năm và phần bổ sung thì giữ nguyên toàn bộ phần tên đó.

Một câu có thể chứa nhiều luật khác nhau; mỗi cặp ghi trên một dòng riêng, định dạng:
ART: số_điều1,số_điều2,... | LAW: tên_đầy_đủ_của_luật

Nếu một luật được nhắc đến nhưng không có Điều nào kèm theo (ví dụ chỉ nói "Nghị quyết số …" mà không dẫn Điều), bạn vẫn xuất dòng đó với phần ART để trống: ART: | LAW: tên_luật.

Chỉ xuất các dòng kết quả, không thêm bất cứ văn bản nào khác.

Dưới đây là 18 ví dụ minh họa (bao gồm cả trường hợp không có trích dẫn nào) để bạn học cách làm đúng.

Ví dụ 1:
Input: khoản 4 Điều 174 Bộ luật Hình sự.
Output: ART: 174 | LAW: Bộ luật Hình sự

Ví dụ 2:
Input: các Điều 271, Điều 272 và Điều 273 Bộ luật tố tụng Dân sự, được chấp nhận xem xét theo thủ tục phúc thẩm.
Output: ART: 271,272,273 | LAW: Bộ luật tố tụng Dân sự

Ví dụ 3:
Input: khoản 1 Điều 308 Bộ luật tố tụng Dân sự: Đề nghị Hội đồng xét xử phúc thẩm Tòa án nhân dân tỉnh Điện Biên giữ nguyên Bản án dân sự sơ thẩm số 04/2025/DS-ST ngày 28/9/2025 của Tòa án nhân dân khu vực 3, tỉnh Điện Biên.
Output: ART: 308 | LAW: Bộ luật tố tụng Dân sự

Ví dụ 4:
Input: các Điều 33, 38, 59 của Luật Hôn nhân và gia đình; các Điều 463, 466, 357, 468 của Bộ luật Dân sự; Nghị quyết số 326/2016/UBTVQH14
Output:
ART: 33,38,59 | LAW: Luật Hôn nhân và gia đình
ART: 463,466,357,468 | LAW: Bộ luật Dân sự
ART: | LAW: Nghị quyết số 326/2016/UBTVQH14

Ví dụ 5:
Input: khoản 2 Điều 468 Bộ luật Dân sự (10%/năm).
Output: ART: 468 | LAW: Bộ luật Dân sự

Ví dụ 6:
Input: Điều 26, 31, 235, 236 Luật đất đai 2024; Điều 115, Điều 158, Điều 159, Điều 165, Điều 166, Điều 175, Điều 176 Bộ luật dân sự; Căn cứ khoản 2 Điều 26 Nghị quyết số 326/2016/UBTVQH ngày 30/12/2016 của Ủy ban thường vụ Quốc hội quy định về án phí, lệ phí Tòa án.
Output:
ART: 26,31,235,236 | LAW: Luật đất đai 2024
ART: 115,158,159,165,166,175,176 | LAW: Bộ luật dân sự
ART: 26 | LAW: Nghị quyết số 326/2016/UBTVQH ngày 30/12/2016 của Ủy ban thường vụ Quốc hội

Ví dụ 7:
Input: khoản 1 Điều 308; Điều 148 của Bộ luật Tố tụng dân sự, Nghị quyết số 326/2016/UBTVQH14 ngày 30/12/2016 quy định về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng phí, án phí Tòa án của Ủy ban thường vụ Quốc hội.
Output:
ART: 308,148 | LAW: Bộ luật Tố tụng dân sự
ART: | LAW: Nghị quyết số 326/2016/UBTVQH14 ngày 30/12/2016 của Ủy ban thường vụ Quốc hội

Ví dụ 13:
Input: khoản 2 Điều 3 Luật số 14/2017/QH14 của Quốc Hội về quản lý, sử dụng vũ khí, vật liệu nổ và công cụ hỗ trợ thì bật lửa hình lựu đạn này không thuộc vũ khí quân dụng.
Output: ART: 3 | LAW: Luật số 14/2017/QH14 của Quốc Hội về quản lý, sử dụng vũ khí, vật liệu nổ và công cụ hỗ trợ

Ví dụ 14:
Input: các điểm (đỉnh thửa): A-B-C-E-F-G-H-K-L-M thể hiện tại Mảnh trích đo địa chính số 36- 2025 do Công ty cổ phần đo đạc N thực hiện ngày 10 tháng 4 năm 2025 kèm theo Bản án.
Output: (không xuất dòng nào)

Ví dụ 15:
Input: Điểm 3: 1244848. Điểm 4: 1244849.
Output: (không xuất dòng nào)

Ví dụ 16 (Nghị định):
Input: "khoản 1 Điều 14 Nghị định số 91/2019/NĐ- CP ngày 19/11/2019 của Chính phủ về xử phạt vi phạm hành chính trong lĩnh vực đất đai."
Output: ART: 14 | LAW: Nghị định số 91/2019/NĐ- CP ngày 19/11/2019 của Chính phủ về xử phạt vi phạm hành chính trong lĩnh vực đất đai
"""

client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

async def label_one(sem, entry_index, law_index, text):
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
            return entry_index, law_index, raw, usage.dict() if usage else None
        except Exception as e:
            return entry_index, law_index, f"ERROR: {str(e)}", None

async def main():
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
            tasks.append(label_one(sem, i, j, law_text))

    results = {}
    completed = 0
    total = len(tasks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            entry_idx, law_idx, raw, usage = await coro
            results[(entry_idx, law_idx)] = (raw, usage)
            completed += 1

            entry = items[entry_idx]
            out.write(json.dumps({
                "file_name": entry.get("file_name", ""),
                "entry_index": entry_idx,
                "law_index": law_idx,
                "input": entry["laws_cited"][law_idx],
                "output": raw,
                "usage": usage
            }, ensure_ascii=False) + "\n")
            out.flush()

            if completed % 1000 == 0:
                print(f"Progress: {completed}/{total}")

    print("Labelling complete. Output written to", OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())