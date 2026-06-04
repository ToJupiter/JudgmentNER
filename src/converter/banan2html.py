#!/usr/bin/env python3

from pathlib import Path
import argparse
import html

import fitz  # PyMuPDF


def convert_pdf_to_absolute_html(
    pdf_path: Path,
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_html = output_dir / f"{pdf_path.stem}_layout.html"

    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <style>",
        "        body {",
        "            background-color: #f0f0f0;",
        "            font-family: 'Times New Roman', Times, serif;",
        "        }",
        "        .page {",
        "            background-color: white;",
        "            position: relative;",
        "            margin: 20px auto;",
        "            border: 1px solid #ccc;",
        "            box-shadow: 0 4px 8px rgba(0,0,0,0.1);",
        "        }",
        "        .text-line {",
        "            position: absolute;",
        "            white-space: nowrap;",
        "            transform-origin: left top;",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
    ]

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_width = page.rect.width
            page_height = page.rect.height

            html_content.append(
                (
                    f"<div class='page' "
                    f"style='width:{page_width}px;height:{page_height}px;' "
                    f"data-page='{page_num}'>"
                )
            )

            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    lx0, ly0, _, _ = line["bbox"]

                    line_text = []
                    font_name = "Times New Roman"
                    font_size = 11.0

                    for span in line.get("spans", []):
                        line_text.append(span["text"])
                        font_name = span["font"]
                        font_size = span["size"]

                    text = "".join(line_text)

                    if not text.strip():
                        continue

                    html_content.append(
                        (
                            "<div class='text-line' "
                            f"style='left:{lx0:.1f}px;"
                            f"top:{ly0:.1f}px;"
                            f"font-size:{font_size:.1f}px;"
                            f"font-family:{html.escape(font_name)};'>"
                            f"{html.escape(text)}"
                            "</div>"
                        )
                    )

            html_content.append("</div>")

    html_content.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"HTML layout exported: {output_html}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PDF into absolutely positioned HTML."
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Input PDF file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for generated HTML",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    convert_pdf_to_absolute_html(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
