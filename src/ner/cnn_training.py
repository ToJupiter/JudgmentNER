import argparse
import csv
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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
AMENDMENT_SUFFIX = r'(?:\s*\(\s*(?:sửa đổi|bổ sung|sửa đổi, bổ sung)[^)]*\))?'

LAW_REF = rf'''
(?:
    {LAW_ABBR}\b
    (?:\s*(?:năm\s*)?\(?\d{{4}}\)?)?
    {AMENDMENT_SUFFIX}
    |
    {LAW_KIND}\b
    [^.;\n]{{0,180}}
    {AMENDMENT_SUFFIX}
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
    {AMENDMENT_SUFFIX}
)
''', FLAGS)

ARTICLE_UNIT_RE = re.compile(rf'''
(?:các\s+)?
{POINT_PART}
{CLAUSE_PART}
{DIEU}\s*{NUM_LIST}
''', FLAGS)

LAW_NAME_RE = re.compile(rf'''
{LAW_REF}
''', FLAGS)

ABBR_MAP = {
    "blhs": "bộ luật hình sự",
    "blds": "bộ luật dân sự",
    "blttds": "bộ luật tố tụng dân sự",
    "bltths": "bộ luật tố tụng hình sự",
    "bltthc": "bộ luật tố tụng hành chính"
}

FEATURE_NAMES = [
    "is_numberish",
    "is_year",
    "is_article_keyword",
    "is_clause_keyword",
    "is_point_keyword",
    "is_law_kind",
    "is_law_abbr",
    "is_connector",
    "is_punct",
    "inside_citation_regex",
    "inside_article_regex",
    "inside_law_regex",
    "gazetteer_law_begin",
    "gazetteer_law_inside",
    "near_article_keyword",
    "near_law_keyword"
]


def normalize_legal_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(điều|khoản|điểm)\s*(?=\d)', r'\1 ', text, flags=re.I)
    text = re.sub(r'\bđê\s*(?=\d)', 'Điều ', text, flags=re.I)
    law_start = r'(?:bộ\s+luật|luật|BLHS|BLDS|BLTTDS|BLTTHS|BLTTHC|nghị\s+quyết|nghị\s+định|thông\s+tư|pháp\s+lệnh|án\s+lệ)'
    text = re.sub(rf';\s*((?:của|thuộc|theo)\s+(?={law_start}\b))', r' \1', text, flags=re.I)
    return text.strip()


def normalize_key(text: str) -> str:
    text = normalize_legal_text(text)
    text = text.lower()
    text = re.sub(r'[“”"\'`]', ' ', text)
    text = re.sub(r'\b(năm)\b', ' ', text)
    text = re.sub(r'[^\w\s/-]', ' ', text, flags=re.U)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def token_clean(token: str) -> str:
    return token.strip('.,;:()[]{}“”"\'`').lower()


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens = []
    spans = []
    for m in re.finditer(r'\w+|[^\w\s]', text, flags=re.U):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def span_to_token_indices(token_spans: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    return [i for i, (s, e) in enumerate(token_spans) if e > start and s < end]


def mark_span(tags: List[str], indices: List[int], label: str):
    if not indices:
        return
    indices = sorted(set(indices))
    for k, idx in enumerate(indices):
        tags[idx] = f"B-{label}" if k == 0 else f"I-{label}"


def find_all_substring_spans(text: str, needle: str) -> List[Tuple[int, int]]:
    out = []
    if not needle:
        return out
    variants = []
    raw = needle.strip()
    variants.append(raw)
    variants.append(raw.rstrip('.,;: '))
    variants.append(normalize_legal_text(raw))
    variants.append(normalize_legal_text(raw).rstrip('.,;: '))
    seen = set()
    for v in variants:
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        start = 0
        lower_text = text.lower()
        lower_v = v.lower()
        while True:
            idx = lower_text.find(lower_v, start)
            if idx < 0:
                break
            out.append((idx, idx + len(v)))
            start = idx + 1
    return sorted(set(out))


def parse_output_raw(output_raw: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    articles = []
    law_ranges = []
    if not output_raw:
        return articles, law_ranges
    art_m = re.search(r'ART:\s*([^|]+)', output_raw, flags=re.I)
    if art_m:
        raw = art_m.group(1).strip()
        if raw:
            for x in re.split(r'[,\s]+', raw):
                x = x.strip()
                if x:
                    articles.append(x)
    law_m = re.search(r'LAW:\s*([0-9]+)\s*-\s*([0-9]+)', output_raw, flags=re.I)
    if law_m:
        a = int(law_m.group(1))
        b = int(law_m.group(2))
        if a <= b:
            law_ranges.append((a, b))
    return articles, law_ranges


def load_law_database(csv_path: Optional[str]) -> List[str]:
    titles = []
    if not csv_path:
        return titles
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first:
                first = False
                if row and row[0].strip().lower() != "title":
                    titles.append(row[0].strip().strip('"'))
                continue
            if row and row[0].strip():
                titles.append(row[0].strip().strip('"'))
    return list(dict.fromkeys(titles))


def law_title_variants(title: str) -> List[List[str]]:
    base = normalize_key(title)
    variants = [base]
    for abbr, full in ABBR_MAP.items():
        if full in base:
            variants.append(base.replace(full, abbr))
    out = []
    for v in variants:
        toks = [t for t in re.findall(r'\w+', v, flags=re.U) if t]
        if len(toks) >= 2:
            out.append(toks)
    return out


def build_law_gazetteer(law_titles: List[str]) -> List[List[str]]:
    items = []
    for title in law_titles:
        items.extend(law_title_variants(title))
    for abbr, full in ABBR_MAP.items():
        items.append([abbr])
        items.append(full.split())
    uniq = []
    seen = set()
    for item in items:
        key = tuple(item)
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    uniq.sort(key=len, reverse=True)
    return uniq


def char_span_mask(text: str, regex: re.Pattern) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in regex.finditer(text)]


def token_inside_any_span(token_spans: List[Tuple[int, int]], char_spans: List[Tuple[int, int]]) -> List[int]:
    mask = [0] * len(token_spans)
    for i, (s, e) in enumerate(token_spans):
        for a, b in char_spans:
            if e > a and s < b:
                mask[i] = 1
                break
    return mask


def gazetteer_token_mask(tokens: List[str], gazetteer: List[List[str]]) -> Tuple[List[int], List[int]]:
    norm_tokens = [normalize_key(t) for t in tokens]
    begin = [0] * len(tokens)
    inside = [0] * len(tokens)
    for item in gazetteer:
        n = len(item)
        if n == 0 or n > len(tokens):
            continue
        for i in range(0, len(tokens) - n + 1):
            if norm_tokens[i:i + n] == item:
                begin[i] = 1
                for j in range(i, i + n):
                    inside[j] = 1
    return begin, inside


def token_distance_flag(tokens: List[str], keywords: set, window: int) -> List[int]:
    positions = [i for i, t in enumerate(tokens) if token_clean(t) in keywords]
    out = [0] * len(tokens)
    for i in range(len(tokens)):
        for p in positions:
            if abs(i - p) <= window:
                out[i] = 1
                break
    return out


def extract_features(text: str, tokens: List[str], token_spans: List[Tuple[int, int]], gazetteer: List[List[str]]) -> List[List[float]]:
    citation_mask = token_inside_any_span(token_spans, char_span_mask(text, CITATION_CANDIDATE_RE))
    article_mask = token_inside_any_span(token_spans, char_span_mask(text, ARTICLE_UNIT_RE))
    law_mask = token_inside_any_span(token_spans, char_span_mask(text, LAW_NAME_RE))
    gaz_begin, gaz_inside = gazetteer_token_mask(tokens, gazetteer)
    near_art = token_distance_flag(tokens, {"điều", "khoản", "điểm"}, 3)
    near_law = token_distance_flag(tokens, {"luật", "bộ", "nghị", "định", "quyết", "thông", "tư", "pháp", "lệnh"}, 4)
    feats = []
    for i, tok in enumerate(tokens):
        c = token_clean(tok)
        low = tok.lower()
        is_numberish = 1 if re.fullmatch(r'\d{1,4}[a-zA-ZđĐ]?', c or "") else 0
        is_year = 1 if re.fullmatch(r'(19|20)\d{2}', c or "") else 0
        is_article_keyword = 1 if c == "điều" else 0
        is_clause_keyword = 1 if c == "khoản" else 0
        is_point_keyword = 1 if c == "điểm" else 0
        is_law_kind = 1 if c in {"luật", "bộ", "nghị", "định", "quyết", "thông", "tư", "pháp", "lệnh", "án", "lệ"} else 0
        is_law_abbr = 1 if low in ABBR_MAP else 0
        is_connector = 1 if c in {"của", "theo", "thuộc", "và", "hoặc", "căn", "cứ", "áp", "dụng", "tại"} else 0
        is_punct = 1 if re.fullmatch(r'[^\w\s]', tok, flags=re.U) else 0
        feats.append([
            is_numberish,
            is_year,
            is_article_keyword,
            is_clause_keyword,
            is_point_keyword,
            is_law_kind,
            is_law_abbr,
            is_connector,
            is_punct,
            citation_mask[i],
            article_mask[i],
            law_mask[i],
            gaz_begin[i],
            gaz_inside[i],
            near_art[i],
            near_law[i]
        ])
    return feats


def select_output_raw_law_span(tokens: List[str], law_ranges: List[Tuple[int, int]]) -> Optional[List[int]]:
    candidates = []
    n = len(tokens)
    for a, b in law_ranges:
        options = [
            (a, b),
            (a - 1, b - 1),
            (a, b - 1),
            (a - 1, b)
        ]
        for s, e in options:
            if 0 <= s <= e < n:
                inds = list(range(s, e + 1))
                text = " ".join(tokens[i].lower() for i in inds)
                score = 0
                for kw in ["luật", "bộ", "nghị", "định", "quyết", "thông", "tư", "pháp", "lệnh", "blhs", "blds", "blttds", "bltths", "bltthc"]:
                    if kw in text:
                        score += 2
                score += len(inds) * 0.01
                candidates.append((score, inds))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def tag_laws_from_citations(text: str, tokens: List[str], token_spans: List[Tuple[int, int]], entry: Dict[str, Any], tags: List[str]) -> int:
    count = 0
    for cit in entry.get("citations", []):
        law = cit.get("law", "")
        spans = find_all_substring_spans(text, normalize_legal_text(law))
        if not spans:
            spans = find_all_substring_spans(text, law)
        if spans:
            best = max(spans, key=lambda x: x[1] - x[0])
            inds = span_to_token_indices(token_spans, best[0], best[1])
            if inds:
                mark_span(tags, inds, "LAW")
                count += 1
    if count == 0:
        _, law_ranges = parse_output_raw(entry.get("output_raw", ""))
        inds = select_output_raw_law_span(tokens, law_ranges)
        if inds:
            mark_span(tags, inds, "LAW")
            count += 1
    return count


def find_article_indices(tokens: List[str], article: str) -> List[List[int]]:
    out = []
    art = token_clean(article)
    clean_tokens = [token_clean(t) for t in tokens]
    for i, c in enumerate(clean_tokens):
        if c != art:
            continue
        if i > 0 and clean_tokens[i - 1] == "điều":
            out.append([i - 1, i])
        else:
            left = max(0, i - 8)
            has_dieu = any(clean_tokens[j] == "điều" for j in range(left, i))
            if has_dieu:
                out.append([i])
    return out


def tag_articles_from_citations(tokens: List[str], entry: Dict[str, Any], tags: List[str]) -> int:
    articles = []
    for cit in entry.get("citations", []):
        for art in cit.get("articles", []):
            if art:
                articles.append(str(art))
    parsed_articles, _ = parse_output_raw(entry.get("output_raw", ""))
    articles.extend(parsed_articles)
    articles = list(dict.fromkeys([a.strip() for a in articles if a.strip()]))
    used = set()
    count = 0
    for art in articles:
        candidates = find_article_indices(tokens, art)
        for inds in candidates:
            key = tuple(inds)
            if key in used:
                continue
            used.add(key)
            mark_span(tags, inds, "ART")
            count += 1
            break
    return count


def make_sample(entry: Dict[str, Any], gazetteer: List[List[str]]) -> Optional[Dict[str, Any]]:
    raw_text = entry.get("input", "")
    if not raw_text:
        return None
    text = normalize_legal_text(raw_text)
    tokens, token_spans = tokenize_with_spans(text)
    if not tokens:
        return None
    tags = ["O"] * len(tokens)
    tag_laws_from_citations(text, tokens, token_spans, entry, tags)
    tag_articles_from_citations(tokens, entry, tags)
    features = extract_features(text, tokens, token_spans, gazetteer)
    return {
        "tokens": tokens,
        "tags": tags,
        "features": features,
        "text": text,
        "file_name": entry.get("file_name", ""),
        "entry_index": entry.get("entry_index", -1),
        "law_index": entry.get("law_index", -1)
    }


def read_jsonl_files(files: List[str]) -> List[Dict[str, Any]]:
    entries = []
    bad = 0
    for fname in files:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    bad += 1
    if bad:
        print(f"[DATA] Bad JSONL lines skipped: {bad}")
    return entries


def dedup_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = {}
    for e in entries:
        key = (e.get("file_name", ""), e.get("entry_index", -1), e.get("law_index", -1), e.get("input", ""))
        if key not in out:
            out[key] = e
    return list(out.values())


def prepare_data(jsonl_files: List[str], law_titles: List[str]) -> List[Dict[str, Any]]:
    gazetteer = build_law_gazetteer(law_titles)
    entries = dedup_entries(read_jsonl_files(jsonl_files))
    samples = []
    pos = 0
    neg = 0
    for entry in entries:
        sample = make_sample(entry, gazetteer)
        if sample is None:
            continue
        has_entity = any(t != "O" for t in sample["tags"])
        if has_entity:
            pos += 1
        else:
            neg += 1
        samples.append(sample)
    print(f"[DATA] Entries: {len(entries)}")
    print(f"[DATA] Samples: {len(samples)}")
    print(f"[DATA] Positive: {pos}")
    print(f"[DATA] Negative: {neg}")
    return samples


def split_by_file(samples: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_file = defaultdict(list)
    for s in samples:
        by_file[s.get("file_name", "")].append(s)
    files = list(by_file.keys())
    random.Random(seed).shuffle(files)
    n = len(files)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_files = set(files[:train_end])
    val_files = set(files[train_end:val_end])
    test_files = set(files[val_end:])
    train = [s for s in samples if s.get("file_name", "") in train_files]
    val = [s for s in samples if s.get("file_name", "") in val_files]
    test = [s for s in samples if s.get("file_name", "") in test_files]
    if not val and train:
        val = train[-max(1, len(train) // 10):]
        train = train[:-len(val)]
    if not test and train:
        test = train[-max(1, len(train) // 10):]
        train = train[:-len(test)]
    return train, val, test


def build_vocab(samples: List[Dict[str, Any]], min_freq: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    word_counter = Counter()
    char_counter = Counter()
    for s in samples:
        for t in s["tokens"]:
            word_counter[t] += 1
            word_counter[t.lower()] += 1
            for ch in t:
                char_counter[ch] += 1
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in word_counter.most_common():
        if c >= min_freq and w not in word2idx:
            word2idx[w] = len(word2idx)
    for ch, c in char_counter.most_common():
        if c >= 1 and ch not in char2idx:
            char2idx[ch] = len(char2idx)
    return word2idx, char2idx


class LegalNERDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], word2idx: Dict[str, int], char2idx: Dict[str, int], tag2idx: Dict[str, int], max_word_len: int):
        self.samples = samples
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.samples)

    def word_id(self, token: str) -> int:
        if token in self.word2idx:
            return self.word2idx[token]
        low = token.lower()
        if low in self.word2idx:
            return self.word2idx[low]
        return self.word2idx["<UNK>"]

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = s["tokens"]
        tags = s["tags"]
        word_ids = [self.word_id(t) for t in tokens]
        char_ids = []
        for t in tokens:
            arr = [self.char2idx.get(ch, self.char2idx["<UNK>"]) for ch in t[:self.max_word_len]]
            arr += [self.char2idx["<PAD>"]] * (self.max_word_len - len(arr))
            char_ids.append(arr)
        tag_ids = [self.tag2idx[t] for t in tags]
        feats = s["features"]
        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(feats, dtype=torch.float),
            torch.tensor(tag_ids, dtype=torch.long),
            len(tokens),
            s["text"]
        )


def collate_fn(batch):
    word_ids, char_ids, feats, tag_ids, lengths, texts = zip(*batch)
    word_pad = pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_pad = pad_sequence(tag_ids, batch_first=True, padding_value=0)
    char_pad = pad_sequence(char_ids, batch_first=True, padding_value=0)
    feat_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(word_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return word_pad, char_pad, feat_pad, tag_pad, lengths, mask, texts


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int, filters: int, kernel_size: int, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(char_emb_dim, filters, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()

    def forward(self, char_ids):
        b, s, w = char_ids.shape
        x = char_ids.reshape(b * s, w)
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv(x))
        x = torch.max(x, dim=-1)[0]
        return x.reshape(b, s, -1)


class LinearCRF(nn.Module):
    def __init__(self, num_tags: int, pad_idx: int):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        with torch.no_grad():
            self.transitions[:, self.pad_idx] = -10000.0
            self.transitions[self.pad_idx, :] = -10000.0
            self.start_transitions[self.pad_idx] = -10000.0
            self.end_transitions[self.pad_idx] = -10000.0

    def forward_score(self, emissions, tags, mask):
        b, s, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, s):
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            score = score + (emit + trans) * mask[:, i]
        lengths = mask.long().sum(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def log_partition(self, emissions, mask):
        b, s, n = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, s):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, i].unsqueeze(1)
            next_score = torch.logsumexp(broadcast_score + self.transitions + broadcast_emit, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        mask = mask.bool()
        return (self.log_partition(emissions, mask) - self.forward_score(emissions, tags, mask)).mean()

    def decode(self, emissions, mask):
        mask = mask.bool()
        b, s, n = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, s):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            best_score, best_path = next_score.max(1)
            score = torch.where(mask[:, i].unsqueeze(1), best_score, score)
            history.append(best_path)
        score = score + self.end_transitions
        best_last = score.argmax(1)
        paths = []
        lengths = mask.long().sum(1)
        for bi in range(b):
            seq_len = lengths[bi].item()
            last = best_last[bi].item()
            path = [last]
            for hist in reversed(history[:seq_len - 1]):
                last = hist[bi][last].item()
                path.append(last)
            path.reverse()
            paths.append(path)
        return paths


class LegalNERModel(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        feature_dim: int,
        tag_size: int,
        word_emb_dim: int,
        char_emb_dim: int,
        char_filters: int,
        feature_proj_dim: int,
        lstm_hidden_dim: int,
        dropout: float,
        pad_idx: int
    ):
        super().__init__()
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=pad_idx)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, char_filters, 3, pad_idx)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_proj_dim),
            nn.ReLU()
        )
        input_dim = word_emb_dim + char_filters + feature_proj_dim
        self.dropout_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_out = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, tag_size)
        self.crf = LinearCRF(tag_size, pad_idx)

    def emissions(self, word_ids, char_ids, feats, lengths):
        w = self.word_emb(word_ids)
        c = self.char_cnn(char_ids)
        f = self.feature_proj(feats)
        x = torch.cat([w, c, f], dim=-1)
        x = self.dropout_in(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=word_ids.size(1))
        out = self.dropout_out(out)
        return self.fc(out)

    def loss(self, word_ids, char_ids, feats, tags, lengths, mask):
        emissions = self.emissions(word_ids, char_ids, feats, lengths)
        return self.crf.neg_log_likelihood(emissions, tags, mask)

    def decode(self, word_ids, char_ids, feats, lengths, mask):
        emissions = self.emissions(word_ids, char_ids, feats, lengths)
        return self.crf.decode(emissions, mask)


def spans_from_tags(tags: List[str], label: str) -> set:
    spans = set()
    start = None
    for i, t in enumerate(tags):
        if t == f"B-{label}":
            if start is not None:
                spans.add((start, i - 1))
            start = i
        elif t == f"I-{label}":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.add((start, i - 1))
                start = None
    if start is not None:
        spans.add((start, len(tags) - 1))
    return spans


def normalize_bio(tags: List[str]) -> List[str]:
    out = []
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            label = t[2:]
            if prev not in {f"B-{label}", f"I-{label}"}:
                out.append(f"B-{label}")
            else:
                out.append(t)
        else:
            out.append(t)
        prev = out[-1]
    return out


def compute_metrics(true_tags_list: List[List[str]], pred_tags_list: List[List[str]]) -> Dict[str, float]:
    labels = ["LAW", "ART"]
    result = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = fp = fn = 0
        for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
            t = spans_from_tags(true_tags, label)
            p = spans_from_tags(normalize_bio(pred_tags), label)
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        result[f"{label.lower()}_prec"] = prec
        result[f"{label.lower()}_rec"] = rec
        result[f"{label.lower()}_f1"] = f1
        result[f"{label.lower()}_tp"] = tp
        result[f"{label.lower()}_fp"] = fp
        result[f"{label.lower()}_fn"] = fn
        total_tp += tp
        total_fp += fp
        total_fn += fn
    result["micro_prec"] = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    result["micro_rec"] = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    result["micro_f1"] = 2 * result["micro_prec"] * result["micro_rec"] / (result["micro_prec"] + result["micro_rec"]) if result["micro_prec"] + result["micro_rec"] else 0.0
    result["avg_f1"] = (result["law_f1"] + result["art_f1"]) / 2
    return result


def evaluate(model, loader, idx2tag, device, name: str):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for word_ids, char_ids, feats, tag_ids, lengths, mask, _ in loader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            feats = feats.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            loss = model.loss(word_ids, char_ids, feats, tag_ids, lengths, mask)
            losses.append(loss.item())
            paths = model.decode(word_ids, char_ids, feats, lengths, mask)
            for i, path in enumerate(paths):
                l = int(lengths[i].item())
                pred = [idx2tag[x] for x in path[:l]]
                true = [idx2tag[int(x)] for x in tag_ids[i][:l].detach().cpu().tolist()]
                all_pred.append(pred)
                all_true.append(true)
    metrics = compute_metrics(all_true, all_pred)
    avg_loss = sum(losses) / max(1, len(losses))
    print(f"[{name}] loss={avg_loss:.4f} micro_f1={metrics['micro_f1']:.4f} avg_f1={metrics['avg_f1']:.4f}")
    print(f"[{name}] LAW P={metrics['law_prec']:.4f} R={metrics['law_rec']:.4f} F1={metrics['law_f1']:.4f} tp={metrics['law_tp']} fp={metrics['law_fp']} fn={metrics['law_fn']}")
    print(f"[{name}] ART P={metrics['art_prec']:.4f} R={metrics['art_rec']:.4f} F1={metrics['art_f1']:.4f} tp={metrics['art_tp']} fp={metrics['art_fp']} fn={metrics['art_fn']}")
    return avg_loss, metrics


def ids_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    tags = normalize_bio(tags)
    out = []
    start = None
    label = None
    for i, t in enumerate(tags + ["O"]):
        if i < len(tags) and t.startswith("B-"):
            if start is not None:
                out.append({"label": label, "start": start, "end": i - 1, "text": " ".join(tokens[start:i])})
            start = i
            label = t[2:]
        elif i < len(tags) and t.startswith("I-"):
            cur = t[2:]
            if start is None:
                start = i
                label = cur
            elif cur != label:
                out.append({"label": label, "start": start, "end": i - 1, "text": " ".join(tokens[start:i])})
                start = i
                label = cur
        else:
            if start is not None:
                out.append({"label": label, "start": start, "end": i - 1, "text": " ".join(tokens[start:i])})
                start = None
                label = None
    return out


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bundle(output_dir: Path, model, config, word2idx, char2idx, tag2idx, law_titles):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "word2idx.json", word2idx)
    save_json(output_dir / "char2idx.json", char2idx)
    save_json(output_dir / "tag2idx.json", tag2idx)
    save_json(output_dir / "law_titles.json", law_titles)


def load_bundle(model_dir: Path, device):
    config = load_json(model_dir / "config.json")
    word2idx = load_json(model_dir / "word2idx.json")
    char2idx = load_json(model_dir / "char2idx.json")
    tag2idx = load_json(model_dir / "tag2idx.json")
    law_titles = load_json(model_dir / "law_titles.json")
    idx2tag = {v: k for k, v in tag2idx.items()}
    model = LegalNERModel(
        len(word2idx),
        len(char2idx),
        len(FEATURE_NAMES),
        len(tag2idx),
        config["word_emb_dim"],
        config["char_emb_dim"],
        config["char_filters"],
        config["feature_proj_dim"],
        config["lstm_hidden_dim"],
        config["dropout"],
        0
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, config, word2idx, char2idx, tag2idx, idx2tag, law_titles


def predict_text(model, text: str, word2idx, char2idx, idx2tag, law_titles, max_word_len: int, device):
    gazetteer = build_law_gazetteer(law_titles)
    text = normalize_legal_text(text)
    tokens, token_spans = tokenize_with_spans(text)
    feats = extract_features(text, tokens, token_spans, gazetteer)
    word_ids = []
    for t in tokens:
        word_ids.append(word2idx.get(t, word2idx.get(t.lower(), word2idx["<UNK>"])))
    char_ids = []
    for t in tokens:
        arr = [char2idx.get(ch, char2idx["<UNK>"]) for ch in t[:max_word_len]]
        arr += [char2idx["<PAD>"]] * (max_word_len - len(arr))
        char_ids.append(arr)
    if not tokens:
        return {"text": text, "tokens": [], "tags": [], "entities": []}
    word_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
    feat_tensor = torch.tensor([feats], dtype=torch.float).to(device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
    mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)
    with torch.no_grad():
        path = model.decode(word_tensor, char_tensor, feat_tensor, lengths, mask)[0]
    tags = [idx2tag[x] for x in path]
    return {
        "text": text,
        "tokens": tokens,
        "tags": normalize_bio(tags),
        "entities": ids_to_entities(tokens, tags)
    }


def write_error_samples(path: Path, samples: List[Dict[str, Any]], model, dataset, idx2tag, device, limit: int):
    if limit <= 0:
        return
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    rows = []
    model.eval()
    seen = 0
    with torch.no_grad():
        for word_ids, char_ids, feats, tag_ids, lengths, mask, texts in loader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            feats = feats.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            paths = model.decode(word_ids, char_ids, feats, lengths, mask)
            for i, path_pred in enumerate(paths):
                l = int(lengths[i].item())
                pred = [idx2tag[x] for x in path_pred[:l]]
                true = [idx2tag[int(x)] for x in tag_ids[i][:l].detach().cpu().tolist()]
                if spans_from_tags(true, "LAW") != spans_from_tags(pred, "LAW") or spans_from_tags(true, "ART") != spans_from_tags(pred, "ART"):
                    rows.append({
                        "text": texts[i],
                        "true": true,
                        "pred": normalize_bio(pred)
                    })
                    seen += 1
                    if seen >= limit:
                        save_json(path, rows)
                        return
    save_json(path, rows)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    law_titles = load_law_database(args.law_csv)
    samples = prepare_data(args.jsonl_files, law_titles)

    if args.max_samples > 0:
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.max_samples]

    train_samples, val_samples, test_samples = split_by_file(samples, args.seed)

    print(f"[SPLIT] Train={len(train_samples)} Val={len(val_samples)} Test={len(test_samples)}")

    word2idx, char2idx = build_vocab(train_samples, args.min_freq)
    tag2idx = {"<PAD>": 0, "O": 1, "B-LAW": 2, "I-LAW": 3, "B-ART": 4, "I-ART": 5}
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_ds = LegalNERDataset(train_samples, word2idx, char2idx, tag2idx, args.max_word_len)
    val_ds = LegalNERDataset(val_samples, word2idx, char2idx, tag2idx, args.max_word_len)
    test_ds = LegalNERDataset(test_samples, word2idx, char2idx, tag2idx, args.max_word_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[DEVICE] {device}")

    model = LegalNERModel(
        len(word2idx),
        len(char2idx),
        len(FEATURE_NAMES),
        len(tag2idx),
        args.word_emb_dim,
        args.char_emb_dim,
        args.char_filters,
        args.feature_proj_dim,
        args.lstm_hidden_dim,
        args.dropout,
        0
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={total_p:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    config = {
        "word_emb_dim": args.word_emb_dim,
        "char_emb_dim": args.char_emb_dim,
        "char_filters": args.char_filters,
        "feature_proj_dim": args.feature_proj_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "dropout": args.dropout,
        "max_word_len": args.max_word_len,
        "feature_names": FEATURE_NAMES
    }

    best = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for word_ids, char_ids, feats, tag_ids, lengths, mask, _ in train_loader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            feats = feats.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss = model.loss(word_ids, char_ids, feats, tag_ids, lengths, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / max(1, len(losses))
        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f}")

        _, val_metrics = evaluate(model, val_loader, idx2tag, device, "VAL")
        scheduler.step(val_metrics["avg_f1"])

        if val_metrics["avg_f1"] > best:
            best = val_metrics["avg_f1"]
            bad_epochs = 0
            save_bundle(output_dir, model, config, word2idx, char2idx, tag2idx, law_titles)
            print(f"[SAVE] best avg_f1={best:.4f}")
        else:
            bad_epochs += 1
            if args.early_stop > 0 and bad_epochs >= args.early_stop:
                print("[STOP] early stopping")
                break

    model, config, word2idx, char2idx, tag2idx, idx2tag, law_titles = load_bundle(output_dir, device)
    evaluate(model, test_loader, idx2tag, device, "TEST")
    write_error_samples(output_dir / "test_errors.json", test_samples, model, test_ds, idx2tag, device, args.error_limit)

    if test_samples:
        demo = test_samples[:min(5, len(test_samples))]
        for s in demo:
            pred = predict_text(model, s["text"], word2idx, char2idx, idx2tag, law_titles, args.max_word_len, device)
            print(json.dumps(pred, ensure_ascii=False, indent=2))


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, config, word2idx, char2idx, tag2idx, idx2tag, law_titles = load_bundle(Path(args.model_dir), device)
    if args.text:
        result = predict_text(model, args.text, word2idx, char2idx, idx2tag, law_titles, config["max_word_len"], device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.input_file:
        rows = []
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    rows.append(predict_text(model, text, word2idx, char2idx, idx2tag, law_titles, config["max_word_len"], device))
        if args.output_file:
            save_json(Path(args.output_file), rows)
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_files", nargs="+")
    parser.add_argument("--law_csv", default=None)
    parser.add_argument("--output_dir", default="artifacts/legal_ner")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--word_emb_dim", type=int, default=96)
    parser.add_argument("--char_emb_dim", type=int, default=32)
    parser.add_argument("--char_filters", type=int, default=48)
    parser.add_argument("--feature_proj_dim", type=int, default=24)
    parser.add_argument("--lstm_hidden_dim", type=int, default=96)
    parser.add_argument("--max_word_len", type=int, default=24)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--error_limit", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--model_dir", default="artifacts/legal_ner")
    parser.add_argument("--text", default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    if args.infer:
        infer(args)
    else:
        if not args.jsonl_files:
            raise ValueError("--jsonl_files is required for training")
        train(args)


if __name__ == "__main__":
    main()