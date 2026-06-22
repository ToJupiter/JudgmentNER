import re
import unicodedata
import argparse
import html

def normalize_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\b(điều|khoản|điểm)\s*(?=\d)', r'\1 ', text, flags=re.I)
    text = re.sub(r'\bđê\s*(?=\d)', 'Điều ', text, flags=re.I)
    law_start = (
        r'(?:bộ\s+luật|luật|BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC|'
        r'nghị\s+quyết|nghị\s+định|thông\s+tư|pháp\s+lệnh)'
    )
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

def markdown_to_highlighted_html(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = normalize_legal_text(text)
    spans = [m.span() for m in CITATION_CANDIDATE_RE.finditer(text)]
    
    result = []
    last = 0
    for s, e in spans:
        result.append(html.escape(text[last:s]))
        result.append(f'<mark style="background-color: #ffcc99;">{html.escape(text[s:e])}</mark>')
        last = e
    result.append(html.escape(text[last:]))
    
    html_content = """<html>
<head><meta charset="utf-8"><title>Regex Candidate Highlight</title>
<style>body{font-family: sans-serif; font-size: 14px; line-height: 1.6;} pre{white-space: pre-wrap;}</style>
</head><body><pre>""" + "".join(result) + "</pre></body></html>"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Highlighted HTML saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input markdown text file")
    parser.add_argument('--output', default="highlighted.html", help="Output HTML file")
    args = parser.parse_args()
    markdown_to_highlighted_html(args.input, args.output)