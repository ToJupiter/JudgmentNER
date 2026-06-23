import re
import csv
import json
from pathlib import Path
from argparse import ArgumentParser

def get_words(text):
    return re.findall(r'\w+', text.lower())

def calculate_f1(words1, words2):
    if not words1 or not words2:
        return 0.0
    set1 = set(words1)
    set2 = set(words2)
    intersection = set1.intersection(set2)
    if not intersection:
        return 0.0
    precision = len(intersection) / len(set2)
    recall = len(intersection) / len(set1)
    return 2 * precision * recall / (precision + recall)

def load_law_titles(csv_path):
    titles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'title' in row and row['title']:
                titles.append(row['title'].strip())
    return titles

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_json', type=Path, required=True)
    parser.add_argument('-l', '--law_csv', type=Path, required=True)
    parser.add_argument('-ov', '--output_valid', type=Path, required=True)
    parser.add_argument('-oi', '--output_invalid', type=Path, required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.4)
    args = parser.parse_args()

    titles = load_law_titles(args.law_csv)
    title_words_list = [get_words(t) for t in titles]
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_data = []
    invalid_data = []
    
    for entry in data:
        file_name = entry['file_name']
        laws_cited = entry['laws_cited']
        
        valid_laws = []
        invalid_laws = []
        
        for law in laws_cited:
            law_words = get_words(law)
            max_f1 = 0.0
            
            for t_words in title_words_list:
                f1 = calculate_f1(t_words, law_words)
                if f1 > max_f1:
                    max_f1 = f1
                    
            if max_f1 >= args.threshold:
                valid_laws.append(law)
            else:
                invalid_laws.append(law)
                
        valid_data.append({
            'file_name': file_name,
            'laws_cited': valid_laws,
            'count': len(valid_laws)
        })
        
        invalid_data.append({
            'file_name': file_name,
            'laws_cited': invalid_laws,
            'count': len(invalid_laws)
        })
        
    with open(args.output_valid, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
        
    with open(args.output_invalid, 'w', encoding='utf-8') as f:
        json.dump(invalid_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()