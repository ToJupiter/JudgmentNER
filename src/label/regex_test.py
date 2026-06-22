import re
import json
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
import polars as pl

def normalize_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\b(điều|khoản|điểm)\s*(?=\d)', r'\1 ', text, flags=re.I)
    text = re.sub(r'\bđê\s*(?=\d)', 'Điều ', text, flags=re.I)
    law_start = r'(?:bộ\s+luật|luật|BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC|nghị\s+quyết|nghị\s+định|thông\s+tư|pháp\s+lệnh)'
    text = re.sub(rf';\s*((?:của|thuộc|theo)\s+(?={law_start}\b))', r' \1', text, flags=re.I)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

FLAGS = re.I | re.U | re.X
NUM = r'\d{1,4}[a-zA-ZđĐ]?'
DIEU = r'điều'

NUM_LIST = rf'''
{NUM}
(?:
    \s*
    (?:,|và|hoặc|\s+(?={DIEU}\b))
    \s*
    (?:{DIEU}\s*)?
    {NUM}
)*
'''

POINT_LIST = r'''
[a-zA-ZđĐ]
(?:
    \s*,\s*
    [a-zA-ZđĐ]
)*
(?:
    \s*
    (?:và|hoặc)
    \s*
    [a-zA-ZđĐ]
)?
'''

POINT_PART = rf'''
(?:điểm\s+{POINT_LIST}\s*)?
'''

CLAUSE_PART = rf'''
(?:khoản\s+(?:{NUM_LIST})?
    (?:\s*,\s*khoản\s+{NUM_LIST})*
\s*)?
'''

LAW_ABBR = r'''
(?:BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC)
'''

LAW_KIND = r'''
(?:Bộ\s+luật|Luật|Nghị\s+quyết|Nghị\s+định|Thông\s+tư|Pháp\s+lệnh|Án\s+lệ)
'''

LAW_REF = rf'''
(?:
    {LAW_ABBR}\b
    (?:\s*(?:năm\s*)?\(?\d{{4}}\)?)?
    |
    {LAW_KIND}\b
    [^.;\n]{{0,180}}
)
'''

CITATION_CANDIDATE_RE = re.compile(rf'''
(?:
    (?:
        (?:căn\s+cứ|áp\s+dụng|theo\s+quy\s+định\s+tại|quy\s+định\s+tại|
           phù\s+hợp\s+(?:với\s+)?quy\s+định\s+tại|
           vi\s+phạm\s+quy\s+định\s+tại|
           được\s+quy\s+định\s+tại|tại|các?)
        \s*[:,-]?\s*
    )?

    (?:các\s+)?
    {POINT_PART}
    {CLAUSE_PART}
    {DIEU}\s*{NUM_LIST}

    (?:
        \s*
        (?:,|;|và|hoặc|\s+(?={DIEU}\b))
        \s*
        (?:
            (?:{POINT_PART}{CLAUSE_PART}{DIEU}\s*)?
            {NUM_LIST}
        )
    )*

    (?:
        \s*
        (?:của|thuộc|theo)?
        \s*
        {LAW_REF}
    )?
    |
    {LAW_KIND}
    \s*
    (?:số|số|số:)?
    \s*
    [\w./-]+
    [^.;\n]{{0,180}}
)
''', FLAGS)

def load_existing_labeled(jsonl_files: list) -> set:
    if not jsonl_files:
        return set()
        
    dfs = []
    for f in jsonl_files:
        try:
            df = pl.read_ndjson(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return set()
        
    df = pl.concat(dfs)
    if "file_name" not in df.columns or "input" not in df.columns:
        return set()
        
    existing = set(
        row 
        for row in df.select(["file_name", "input"]).iter_rows()
    )
    print(f"[DEDUP] Loaded {len(existing)} existing labeled entries.")
    return existing

def is_valid_candidate(text: str) -> bool:
    if len(text.split()) <= 2:
        return False
    if "luật sư" in text.lower():
        return False
    return True

def test_regex(folder_name: Path, output_file: Path, jsonl_files: list):
    existing_set = load_existing_labeled(jsonl_files)
    results = []
    total = 0
    
    for markdown_file in folder_name.glob('*.md'):
        with open(markdown_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        content = normalize_legal_text(content)
        
        matches = [m.group(0).strip() for m in CITATION_CANDIDATE_RE.finditer(content)]
        
        laws_list = []
        for match in matches:
            match_clean = re.sub(r"\s+", " ", match).strip()
            
            if not is_valid_candidate(match_clean):
                continue
                
            if (markdown_file.name, match_clean) in existing_set:
                continue
                
            laws_list.append(match_clean)
            
        total += len(laws_list)
        results.append({
            "file_name": markdown_file.name,
            "laws_cited": laws_list,
            "count": len(laws_list)
        })
    
    with open(output_file, "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)
    
    return total

def main():
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder_name', type=Path, required=True, help='Folder name to test regex on')
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help='Output JSON file path')
    parser.add_argument('--jsonl_files', nargs='+', default=[], help='Existing JSONL files to deduplicate against')
    args = parser.parse_args()

    total_output = test_regex(args.folder_name, args.output_folder, args.jsonl_files)
    print(f"Total new candidates for DeepSeek: {total_output}")

if __name__ == "__main__":
    main()