import json
import sys

def tokenize_and_index(text):
    tokens = text.split()
    indexed = " ".join(str(i+1) + "#" + tok for i, tok in enumerate(tokens))
    return indexed, tokens

def main():
    input_file = "banan_txt/output_json/june6_laws.json"
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        sys.exit(1)

    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    for i, entry in enumerate(items):
        print(f"=== Entry {i} (file: {entry.get('file_name', 'unknown')}) ===")
        for j, law_text in enumerate(entry.get("laws_cited", [])):
            indexed_text, _ = tokenize_and_index(law_text)
            print(f"--- Law {j} ---")
            print(f"Original: {law_text}")
            print(f"Indexed : {indexed_text}")
            print()

if __name__ == "__main__":
    main()