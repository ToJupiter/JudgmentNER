import json
import re
import sqlite3
from pathlib import Path
import logging
import argparse
from typing import List, Tuple

import polars as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def json_output_to_parquet(input_json: Path, output_parquet: Path, labeled_output: list = [], unlabeled_output: list = []) -> Tuple[List, List]:
    with open(input_json, "r", encoding='utf-8') as json_input:
        for line_number, line in enumerate(json_input, start=1):
            line = line.strip()
            if not line:
                continue
            else:
                try:
                    line_obj = json.loads(line)
                    if line_obj.get("citations"):
                        labeled_output.append(line_obj)
                    else:
                        unlabeled_output.append(line_obj)
                except Exception as e:
                    print(f"Exception happened: {e}")

    logger.info("Size of labeled_output: %d", len(labeled_output))
    logger.info("Size of unlabeled output: %d", len(unlabeled_output))

    return labeled_output, unlabeled_output

def load_and_filter_titles(csv_path: Path) -> pl.DataFrame:
    titles_df = pl.read_csv(csv_path)
    titles = titles_df['title'].cast(pl.Utf8).to_list()
    keywords = ["BLHS", "BLTTHS", "BLDS", "BLTTDS", "nghị quyết", "bộ luật"]

    all_patterns = titles + keywords
    pattern = '|'.join(re.escape(p) for p in all_patterns)
    
    logger.info(f"Loaded {len(titles)} titles from CSV")
    logger.info(f"Filtering with {len(all_patterns)} total patterns")
    return pattern

def filter_unlabeled_by_title(unlabeled_df: pl.DataFrame, csv_path: Path) -> pl.DataFrame:
    pattern = load_and_filter_titles(csv_path)
    if 'input' not in unlabeled_df.columns:
        logger.error("'input' column not found in unlabeled DataFrame")
        return pl.DataFrame()
    
    filtered_df = unlabeled_df.filter(
        pl.col('input').str.contains(f'(?i){pattern}')
    )
    
    logger.info(f"Original unlabeled rows: {len(unlabeled_df)}")
    logger.info(f"Filtered unlabeled rows: {len(filtered_df)}")
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument('-i', '--input', help="Input path to JSONL", required=True)
    parser.add_argument('-o', '--output', help="Output path to parquet", required=True)
    args = parser.parse_args()

    label, unlabeled = json_output_to_parquet(input_json=args.input, output_parquet=args.output)
    label_df = pl.DataFrame(label)
    label_df.write_parquet(args.output + "/labeled.parquet")
    unlabeled_df = pl.DataFrame(unlabeled)
    unlabeled_df.write_parquet(args.output + "/unlabeled.parquet")

    filtered_unlabel = filter_unlabeled_by_title(unlabeled_df=unlabeled_df, csv_path="output_banan_async/texts/law_title.csv")
    filtered_unlabel.write_parquet(args.output + "/unlabeled_filtered.parquet")

if __name__ == "__main__":    
    main()


                    
