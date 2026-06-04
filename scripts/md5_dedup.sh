#!/usr/bin/env bash

declare -A seen
pdf_path="$1"
echo "Usage: $0 <folder_path>"

if [ ! -d "$pdf_path" ]; then
    echo "Error: $pdf_path is not a directory"
    exit 1
fi

find "$pdf_path" -type f -iname "*.pdf" -print0 |
while IFS= read -r -d '' file; do
    hash=$(md5sum "$file" | cut -d' ' -f1)

    if [[ -n "${seen[$hash]}" ]]; then
        echo "Duplicate:"
        echo "  Keep:   ${seen[$hash]}"
        echo "  Delete: $file"
        rm "$file"
    else
        seen[$hash]="$file"
    fi
done