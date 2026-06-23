import re
import json
import unicodedata
import polars as pl
from argparse import ArgumentParser
from pathlib import Path

def normalize_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
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

POINT_PART = rf'(?:điểm\s+{POINT_LIST}\s*)?'
CLAUSE_PART = rf'(?:khoản\s+(?:{NUM_LIST})?(?:\s*,\s*khoản\s+{NUM_LIST})*\s*)?'

LAW_ABBR = r'(?:BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC)'
LAW_KIND = r'(?:Bộ\s+luật|Luật|Nghị\s+quyết|Nghị\s+định|Thông\s+tư|Pháp\s+lệnh|Án\s+lệ)'

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
    (?:số|số|số:)?
    \s*
    [\w./-]+
    [^.;\n]{{0,180}}
)
''', FLAGS)

def load_existing_labeled(jsonl_files):
    if not jsonl_files:
        return {}
        
    dfs = []
    for f in jsonl_files:
        try:
            df = pl.read_ndjson(f)
            
            for col in ["file_name", "input", "citations"]:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))
                    
            df = df.select(["file_name", "input", "citations"])
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return {}
        
    df = pl.concat(dfs)
    
    df = df.with_columns(
        pl.when(pl.col("citations").is_not_null())
          .then(pl.col("citations").cast(pl.Utf8))
          .otherwise("")
          .alias("cit_str")
    )
    df = df.with_columns(
        (pl.col("cit_str").str.lengths() > 2).alias("has_citations")
    )
    
    labeled_df = df.group_by(["file_name", "input"]).agg(
        pl.col("has_citations").max().alias("is_labeled")
    ).filter(pl.col("is_labeled"))
    
    labeled_dict = {}
    for row in labeled_df.iter_rows():
        fname, inp = row[0], row[1]
        if fname not in labeled_dict:
            labeled_dict[fname] = set()
        labeled_dict[fname].add(inp)
        
    return labeled_dict

def test_regex(folder_name: Path, output_file: Path, jsonl_files: list, total: int):
    results = []
    
    print("Loading existing labeled data...")
    labeled_dict = load_existing_labeled(jsonl_files)
    print(f"Loaded labeled data for {len(labeled_dict)} files.")
    
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
            
        labeled_inputs = labeled_dict.get(markdown_file.name, set())
        unlabeled_matches = []
        
        for m in filtered_matches:
            if m in labeled_inputs:
                continue
                
            is_labeled = any(m in L or L in m for L in labeled_inputs)
            if not is_labeled:
                unlabeled_matches.append(m)
                
        unique_matches = list(dict.fromkeys(unlabeled_matches))
        
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
    parser.add_argument('--jsonl_files', nargs='*', default=[], help='List of existing JSONL files to deduplicate against')
    args = parser.parse_args()

    total_output = test_regex(args.folder_name, args.output_folder, args.jsonl_files, total=total_law_list)
    print(f"Total new cited: {total_output}")

if __name__ == "__main__":
    main()