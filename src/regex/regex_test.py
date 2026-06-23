import re
import json
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

def normalize_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\b(điều|khoản|điểm)\s*(?=\d)', r'\1 ', text, flags=re.I)
    text = re.sub(r'\bđê\s*(?=\d)', 'Điều ', text, flags=re.I)
    
    law_start = (
        r'(?:bộ\s+luật|luật|BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC|'
        r'nghị\s+quyết|nghị\s+định|thông\s+tư|pháp\s+lệnh)'
    )
    text = re.sub(
        rf';\s*((?:của|thuộc|theo)\s+(?={law_start}\b))',
        r' \1',
        text,
        flags=re.I
    )
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

POINT_PART = rf'(?:điểm\s+{POINT_LIST}\s*)?'
CLAUSE_PART = rf'(?:khoản\s+(?:{NUM_LIST})?(?:\s*,\s*khoản\s+{NUM_LIST})*\s*)?'

LAW_ABBR = r'(?:BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC)'
LAW_KIND = r'(?:Bộ\s+luật|Luật|Nghị\s+quyết|Nghị\s+định|Thông\s+tư|Pháp\s+lệnh|Án\s+lệ)'

AMENDMENT_SUFFIX = r'(?:\s*\(\s*(?:sửa đổi|bổ sung|sửa đổi, bổ sung)[^)]*\))?'

LAW_REF = rf'''
(?:
    {LAW_ABBR}\b
    (?:\s*(?:năm\s*)?\(?\d{{4}}\)?)?
    {AMENDMENT_SUFFIX}
    |
    {LAW_KIND}\b
    [^.;\n]{{0,180}}
    {AMENDMENT_SUFFIX}
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
    (?:số|số|số:)?
    \s*
    [\w./-]+
    [^.;\n]{{0,180}}
    {AMENDMENT_SUFFIX}
)
''', FLAGS)

def test_regex(folder_name: Path, output_file: Path, total: int):
    results = []
    
    for markdown_file in folder_name.glob('*.md'):
        with open(markdown_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        norm_content = normalize_legal_text(content)
        
        raw_matches = [
            re.sub(r"\s+", " ", m.group(0)).strip() 
            for m in CITATION_CANDIDATE_RE.finditer(norm_content)
        ]
        
        filtered_matches = []
        for m in raw_matches:
            if len(m.split()) <= 2:
                continue
            if "luật sư" in m.lower():
                continue
            filtered_matches.append(m)
            
        unique_matches = list(dict.fromkeys(filtered_matches))
        
        total += len(unique_matches)
        
        results.append({
            "file_name": markdown_file.name,
            "laws_cited": unique_matches,
            "count": len(unique_matches)
        })
        
    with open(output_file, "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)
        
    return total

def main():
    total_law_list: int = 0
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder_name', type=Path, required=True, help='Folder name to test regex on')
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help='Output JSON file path')
    args = parser.parse_args()

    total_output = test_regex(args.folder_name, args.output_folder, total=total_law_list)
    print(f"Total cited: {total_output}")

if __name__ == "__main__":
    main()