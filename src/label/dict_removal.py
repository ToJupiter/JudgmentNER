import json

with open("banan_txt/output_json/june6_laws_filtered.json", "r", encoding='utf-8') as law_list:
    law_list_inmem = json.load(law_list)

target = "regex_test_banan.md"
accepted_data = [d for d in law_list_inmem if d["file_name"] != target]
with open("banan_txt/output_json/june6_laws_filtered.json", "w", encoding='utf-8') as output_file:
    json.dump(accepted_data, output_file, ensure_ascii=False, indent=4)