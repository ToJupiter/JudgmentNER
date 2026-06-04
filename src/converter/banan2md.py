#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF

SPACE_WIDTH_ESTIMATE = 6.0


def convert_pdf_to_reserved_text(pdf_path: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = pdf_path.stem

    json_layout_data = []
    markdown_lines = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_record = {"page": page_num, "elements": []}
            markdown_lines.append("")

            page_dict = page.get_text("dict")
            blocks = sorted(
                page_dict.get("blocks", []),
                key=lambda block: block["bbox"][1],
            )

            for block in blocks:
                if block.get("type") != 0:
                    continue

                lines = sorted(
                    block.get("lines", []),
                    key=lambda line: line["bbox"][1],
                )

                for line in lines:
                    spans = sorted(
                        line.get("spans", []),
                        key=lambda span: span["bbox"][0],
                    )

                    current_line = []
                    last_x1 = None

                    for span in spans:
                        text = span["text"]

                        if not text.strip():
                            continue

                        x0, y0, x1, y1 = span["bbox"]

                        page_record["elements"].append(
                            {
                                "text": text.strip(),
                                "bbox": [
                                    round(x0, 1),
                                    round(y0, 1),
                                    round(x1, 1),
                                    round(y1, 1),
                                ],
                                "font": span["font"],
                                "size": round(span["size"], 1),
                            }
                        )

                        if last_x1 is None:
                            spaces = int(x0 / SPACE_WIDTH_ESTIMATE)
                        else:
                            spaces = max(
                                0,
                                int((x0 - last_x1) / SPACE_WIDTH_ESTIMATE),
                            )

                        current_line.append(" " * spaces)
                        current_line.append(text)

                        last_x1 = x1

                    line_text = "".join(current_line)

                    if line_text.strip():
                        markdown_lines.append(line_text)

            json_layout_data.append(page_record)

    json_output = output_dir / f"{base_name}_layout.json"
    md_output = output_dir / f"{base_name}_visual.md"

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(json_layout_data, f, ensure_ascii=False, indent=2)

    with open(md_output, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print(f"Layout JSON: {json_output}")
    print(f"Visual Markdown: {md_output}")

def folder_iteration(pdf_path: Path, output_dir: Path | None = None) -> None:
    if pdf_path.is_dir():
        for pdf_file in pdf_path.glob("*.pdf"):
            convert_pdf_to_reserved_text(pdf_file, output_dir)
    else:
        convert_pdf_to_reserved_text(pdf_path, output_dir)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract PDF text with geometric layout information."
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
        help="Directory for generated files",
    )

    parser.add_argument(
        "-f",
        "--folder-iteration",
        action="store_true",
        help="Process all PDF files in the specified directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.folder_iteration:
        folder_iteration(args.pdf, args.output_dir)
    else:
        convert_pdf_to_reserved_text(args.pdf, args.output_dir)

if __name__ == "__main__":
    main()
