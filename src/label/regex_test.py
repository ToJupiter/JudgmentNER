import re
import json
from argparse import ArgumentParser
from pathlib import Path

def test_regex(folder_name: Path, output_file: Path, total: int):
    all_laws_cited = []
    results = []
    
    for markdown_file in folder_name.glob('*.md'):
        with open(markdown_file, 'r', encoding='utf-8') as file:
            content = file.read()
            laws_list = [
                re.sub(r"\s+", " ", match).strip()
                for match in re.findall(
                    r'(?:các\s+)?(?:Điều|khoản|điểm)\s+[\d,\s]+(?:và\s+[\d,\s]+)?[^.]*\.',
                    content,
                    re.IGNORECASE
                )
            ]
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
    total_law_list: int = 0
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder_name', type=Path, required=True, help='Folder name to test regex on')
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help='Output JSON file path')
    args = parser.parse_args()

    total_output = test_regex(args.folder_name, args.output_folder, total=total_law_list)
    print(f"Total cited: {total_output}")

if __name__ == "__main__":
    main()