import re
from argparse import ArgumentParser
from pathlib import Path

def test_regex(folder_name: Path, output_file: Path, total: int):
    all_laws_cited = []
    for markdown_file in folder_name.glob('*.md'):
        with open(markdown_file, 'r') as file:
            content = file.read()
            laws_list = re.findall(
                r'(?:các\s+)?(?:Điều|khoản|điểm)\s+[\d,\s]+(?:và\s+[\d,\s]+)?[^.]*\.',
                content,
                re.IGNORECASE
            )
            total += len(laws_list)
            lines_of_laws = "\n".join(laws_list)
            all_laws_cited.append(lines_of_laws)

        with open(output_file, "a", encoding='utf-8') as file_append:
            file_append.write(f"File: {markdown_file.name}\n")
            file_append.write(lines_of_laws + "\n\n")
    return total
            
def main():
    total_law_list: int = 0
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder_name', type=Path, required=True, help='Folder name to test regex on')
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help='Output filename to test regex on')
    args = parser.parse_args()

    total_output = test_regex(args.folder_name, args.output_folder, total=total_law_list)
    print(f"Total cited: {total_output}")

if __name__ == "__main__":
    main()