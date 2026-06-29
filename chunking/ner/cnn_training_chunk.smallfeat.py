import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MARKER_ORDER = ["preamble", "noi_dung_vu_an", "dai_dien_vks", "nhan_dinh", "quyet_dinh"]
TAG_LIST = ["<PAD>", "O", "preamble", "noi_dung_vu_an", "dai_dien_vks", "nhan_dinh", "quyet_dinh"]
CONF_RANK = {"low": 1, "medium": 2, "high": 3}

MARKER_PATTERNS = {
    "preamble": [
        r"^\s*#{0,6}\s*(?:\d+[.)-]\s*)?NHÂN\s+DANH\b.*$",
        r"^\s*#{0,6}\s*(?:\d+[.)-]\s*)?NHAN\s+DANH\b.*$",
        r"NƯỚC\s+CỘNG\s+H[ÒO]A\s+X[ÃA]\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM",
        r"NUOC\s+CONG\s+HOA\s+XA\s+HOI\s+CHU\s+NGHIA\s+VIET\s+NAM"
    ],
    "noi_dung_vu_an": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NỘI\s+DUNG\s+VỤ\s+ÁN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NOI\s+DUNG\s+VU\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NỘI\s+DUNG\s+BẢN\s+ÁN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NOI\s+DUNG\s+BAN\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*THEO\s+C[ÁA]C\s+T[ÀA]I\s+LIỆU.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QU[ÁA]\s+TR[ÌI]NH\s+(?:ĐIỀU\s+TRA|GIẢI\s+QUYẾT).*$"
    ],
    "dai_dien_vks": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*ĐẠI\s+DIỆN\s+VIỆN\s+KIỂM\s+S[ÁA]T.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*DAI\s+DIEN\s+VIEN\s+KIEM\s+SAT.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*ĐẠI\s+DIỆN\s+VKS.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*KIỂM\s+S[ÁA]T\s+VI[ÊE]N.*(?:ĐỀ\s+NGHỊ|PH[ÁA]T\s+BIỂU).*$",
        r"VIỆN\s+KIỂM\s+S[ÁA]T.*(?:ĐỀ\s+NGHỊ|PH[ÁA]T\s+BIỂU|QUAN\s+ĐIỂM)"
    ],
    "nhan_dinh": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHẬN\s+ĐỊNH\s+CỦA\s+T[ÒO]A\s+[ÁA]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHAN\s+DINH\s+CUA\s+TOA\s+AN\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*NHẬN\s+ĐỊNH\s+CỦA\s+HỘI\s+ĐỒNG\s+X[ÉE]T\s+XỬ\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*HỘI\s+ĐỒNG\s+X[ÉE]T\s+XỬ\s+NHẬN\s+ĐỊNH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*HĐXX\s+NHẬN\s+ĐỊNH\s*:?.*$"
    ],
    "quyet_dinh": [
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QUYẾT\s+ĐỊNH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*QUYET\s+DINH\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*V[ÌI]\s+C[ÁA]C\s+LẼ\s+TR[ÊE]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*V[ÌI]\s+C[ÁA]C\s+L[ẼE]\s+TR[ÊE]N\s*:?.*$",
        r"^\s*#{0,6}\s*(?:[IVX]+|\d+)?[.)-]?\s*TUY[ÊE]N\s+XỬ\s*:?.*$"
    ]
}

COMPILED_PATTERNS = {k: [re.compile(p, flags=re.I | re.U) for p in v] for k, v in MARKER_PATTERNS.items()}
FEATURE_NAMES = [
    "pos_ratio", "pos_from_end", "pos_bucket_00_05", "pos_bucket_05_15", "pos_bucket_15_35", "pos_bucket_35_60", "pos_bucket_60_80", "pos_bucket_80_100",
    "log_chars", "log_words", "upper_ratio", "digit_ratio", "punct_ratio", "space_ratio", "has_colon", "has_semicolon", "has_period", "has_comma",
    "starts_hash", "starts_number", "starts_roman", "starts_bracket_number", "is_short_40", "is_short_80", "is_short_120", "title_like", "all_caps", "mostly_lower",
    "regex_preamble", "regex_noi_dung_vu_an", "regex_dai_dien_vks", "regex_nhan_dinh", "regex_quyet_dinh", "regex_any",
    "kw_preamble", "kw_noi_dung", "kw_vks", "kw_nhan_dinh", "kw_quyet_dinh", "kw_toa_an", "kw_hdxx", "kw_bi_cao", "kw_nguyen_don", "kw_khang_cao", "kw_xet_xu", "kw_can_cu", "kw_luat",
    "negative_luat_su", "negative_quyet_dinh_in_sentence", "negative_nhan_dinh_in_sentence", "negative_vks_in_sentence",
    "line_is_empty_like", "has_markdown_table", "has_ocr_noise", "accentless_marker_hit",
    "near_top", "near_middle", "near_late", "near_end",
    "prev_regex_any", "next_regex_any", "prev_title_like", "next_title_like", "prev_blank_like", "next_blank_like",
    "window_regex_preamble", "window_regex_noi_dung_vu_an", "window_regex_dai_dien_vks", "window_regex_nhan_dinh", "window_regex_quyet_dinh"
]


def safe_text(x: Any) -> str:
    return x if isinstance(x, str) else ""


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", safe_text(line))
    line = line.replace("\ufeff", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_lines(text: str) -> list[str]:
    return [normalize_line(x) for x in safe_text(text).splitlines()]


def compact_nonempty_lines(lines: list[str]) -> list[str]:
    return [x for x in lines if x.strip()]


def tokenize_words(line: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_line(line), flags=re.U)


def line_letters(line: str) -> list[str]:
    return [c for c in line if c.isalpha()]


def upper_ratio(line: str) -> float:
    letters = line_letters(line)
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.upper() == c) / len(letters)


def title_like(line: str) -> float:
    x = normalize_line(line)
    if not x:
        return 0.0
    if len(x) <= 90:
        return 1.0
    if x.startswith("#"):
        return 1.0
    if upper_ratio(x) >= 0.55:
        return 1.0
    return 0.0


def regex_flags(line: str) -> dict[str, float]:
    return {k: 1.0 if any(p.search(line) for p in patterns) else 0.0 for k, patterns in COMPILED_PATTERNS.items()}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", safe_text(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")


def plain_line(line: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(normalize_line(line)).lower()).strip()


def marker_plain_flags(line: str) -> dict[str, float]:
    y = plain_line(line)
    return {
        "preamble": 1.0 if "nhan danh" in y or "nuoc cong hoa xa hoi chu nghia viet nam" in y else 0.0,
        "noi_dung_vu_an": 1.0 if "noi dung vu an" in y or "noi dung ban an" in y or "theo cac tai lieu" in y or "qua trinh dieu tra" in y else 0.0,
        "dai_dien_vks": 1.0 if "dai dien vien kiem sat" in y or "kiem sat vien" in y or "vien kiem sat" in y or " vks" in f" {y}" else 0.0,
        "nhan_dinh": 1.0 if "nhan dinh cua toa an" in y or "nhan dinh cua hoi dong xet xu" in y or "hdxx nhan dinh" in y else 0.0,
        "quyet_dinh": 1.0 if "quyet dinh" in y or "vi cac le tren" in y or "tuyen xu" in y else 0.0
    }


def extract_line_features(line: str, index: int, total: int) -> list[float]:
    x = normalize_line(line)
    y = x.lower()
    z = plain_line(x)
    chars = len(x)
    words = tokenize_words(x)
    word_count = len(words)
    punct = sum(1 for c in x if not c.isalnum() and not c.isspace())
    digits = sum(1 for c in x if c.isdigit())
    spaces = sum(1 for c in x if c.isspace())
    pos = index / max(1, total - 1)
    regs = regex_flags(x)
    plain_regs = marker_plain_flags(x)
    regex_any = 1.0 if any(v > 0 for v in regs.values()) else 0.0
    return [
        pos, 1.0 - pos,
        1.0 if pos <= 0.05 else 0.0,
        1.0 if 0.05 < pos <= 0.15 else 0.0,
        1.0 if 0.15 < pos <= 0.35 else 0.0,
        1.0 if 0.35 < pos <= 0.60 else 0.0,
        1.0 if 0.60 < pos <= 0.80 else 0.0,
        1.0 if pos > 0.80 else 0.0,
        min(1.0, math.log1p(chars) / 7.0),
        min(1.0, math.log1p(word_count) / 5.0),
        upper_ratio(x),
        digits / max(1, chars),
        punct / max(1, chars),
        spaces / max(1, chars),
        1.0 if ":" in x else 0.0,
        1.0 if ";" in x else 0.0,
        1.0 if "." in x else 0.0,
        1.0 if "," in x else 0.0,
        1.0 if x.startswith("#") else 0.0,
        1.0 if re.match(r"^\s*\d+[.)-]", x, flags=re.I) else 0.0,
        1.0 if re.match(r"^\s*[IVX]+[.)-]", x, flags=re.I) else 0.0,
        1.0 if re.match(r"^\s*\[?\d+(?:\.\d+)*\]", x, flags=re.I) else 0.0,
        1.0 if chars <= 40 else 0.0,
        1.0 if chars <= 80 else 0.0,
        1.0 if chars <= 120 else 0.0,
        title_like(x),
        1.0 if upper_ratio(x) >= 0.6 else 0.0,
        1.0 if upper_ratio(x) <= 0.12 and chars > 30 else 0.0,
        regs["preamble"], regs["noi_dung_vu_an"], regs["dai_dien_vks"], regs["nhan_dinh"], regs["quyet_dinh"], regex_any,
        max(plain_regs["preamble"], 1.0 if "nhân danh" in y or "nhan danh" in y else 0.0),
        max(plain_regs["noi_dung_vu_an"], 1.0 if "nội dung" in y or "noi dung" in y or "theo các tài liệu" in y else 0.0),
        max(plain_regs["dai_dien_vks"], 1.0 if "viện kiểm sát" in y or "vks" in y or "kiểm sát viên" in y else 0.0),
        max(plain_regs["nhan_dinh"], 1.0 if "nhận định" in y or "nhan dinh" in y or "hđxx" in y else 0.0),
        max(plain_regs["quyet_dinh"], 1.0 if "quyết định" in y or "quyet dinh" in y or "tuyên xử" in y or "vì các lẽ trên" in y else 0.0),
        1.0 if "tòa án" in y or "toa an" in z else 0.0,
        1.0 if "hđxx" in y or "hoi dong xet xu" in z else 0.0,
        1.0 if "bị cáo" in y or "bi cao" in z else 0.0,
        1.0 if "nguyên đơn" in y or "nguyen don" in z else 0.0,
        1.0 if "kháng cáo" in y or "khang cao" in z else 0.0,
        1.0 if "xét xử" in y or "xet xu" in z else 0.0,
        1.0 if "căn cứ" in y or "can cu" in z else 0.0,
        1.0 if "luật" in y or "bo luat" in z or " luat" in f" {z}" else 0.0,
        1.0 if "luật sư" in y or "luat su" in z else 0.0,
        1.0 if ("quyết định" in y or "quyet dinh" in z) and chars > 120 else 0.0,
        1.0 if ("nhận định" in y or "nhan dinh" in z) and chars > 120 else 0.0,
        1.0 if ("viện kiểm sát" in y or "vien kiem sat" in z) and chars > 160 else 0.0,
        1.0 if chars <= 2 else 0.0,
        1.0 if "|" in x and re.search(r"\|.*\|", x) else 0.0,
        1.0 if len(re.findall(r"[^\w\s.,;:()/-]", x, flags=re.U)) >= 3 else 0.0,
        1.0 if any(v > 0 for v in plain_regs.values()) else 0.0,
        1.0 if pos <= 0.2 else 0.0,
        1.0 if 0.2 < pos < 0.78 else 0.0,
        1.0 if pos >= 0.45 else 0.0,
        1.0 if pos >= 0.72 else 0.0
    ]


def contextualize_features(lines: list[str], base_features: list[list[float]]) -> list[list[float]]:
    out = []
    n = len(lines)
    regs = [regex_flags(normalize_line(x)) for x in lines]
    titles = [title_like(x) for x in lines]
    blanks = [1.0 if not normalize_line(x) else 0.0 for x in lines]
    for i, feats in enumerate(base_features):
        prev_regs = regs[i - 1] if i > 0 else {k: 0.0 for k in MARKER_ORDER}
        next_regs = regs[i + 1] if i + 1 < n else {k: 0.0 for k in MARKER_ORDER}
        window = []
        for marker_type in MARKER_ORDER:
            value = regs[i][marker_type]
            if i > 0:
                value = max(value, prev_regs[marker_type] * 0.65)
            if i + 1 < n:
                value = max(value, next_regs[marker_type] * 0.65)
            window.append(value)
        out.append(feats + [
            1.0 if i > 0 and any(prev_regs.values()) else 0.0,
            1.0 if i + 1 < n and any(next_regs.values()) else 0.0,
            titles[i - 1] if i > 0 else 0.0,
            titles[i + 1] if i + 1 < n else 0.0,
            blanks[i - 1] if i > 0 else 1.0,
            blanks[i + 1] if i + 1 < n else 1.0,
            *window
        ])
    return out


def load_label_jsonl(files: list[str]) -> dict[tuple[str, int], tuple[str, int]]:
    labels = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                file_name = safe_text(row.get("file_name"))
                line_index = int(row.get("line_index", -1))
                label = int(row.get("label", 0) or 0)
                marker_type = safe_text(row.get("marker_type"))
                conf = CONF_RANK.get(safe_text(row.get("confidence")).lower(), 0)
                if not file_name or line_index < 0:
                    continue
                key = (file_name, line_index)
                value = marker_type if label == 1 and marker_type in MARKER_ORDER else "O"
                old = labels.get(key)
                if old is None or conf >= old[1]:
                    labels[key] = (value, conf)
    return labels


def build_document_sample(path: Path, label_map: dict[tuple[str, int], tuple[str, int]], require_label: bool) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = compact_nonempty_lines(split_lines(text))
    if not lines:
        return None
    has_any = any((path.name, i) in label_map for i in range(len(lines)))
    if require_label and not has_any:
        return None
    tags = []
    base_feats = []
    words = []
    for i, line in enumerate(lines):
        tags.append(label_map.get((path.name, i), ("O", 0))[0])
        base_feats.append(extract_line_features(line, i, len(lines)))
        words.append(tokenize_words(line))
    feats = contextualize_features(lines, base_feats)
    return {"file_name": path.name, "lines": lines, "words": words, "features": feats, "tags": tags}


def load_samples(markdown_folder: Path, label_files: list[str], use_all_markdown: bool) -> list[dict[str, Any]]:
    print(f"[*] Loading labels from {len(label_files)} files...")
    label_map = load_label_jsonl(label_files)
    print(f"[*] Found {len(label_map)} total labeled line entries.")
    samples = []
    md_files = sorted(markdown_folder.glob("*.md"))
    print(f"[*] Scanning {len(md_files)} markdown files from {markdown_folder}...")
    for path in tqdm(md_files, desc="Loading documents"):
        s = build_document_sample(path, label_map, require_label=not use_all_markdown)
        if s is not None:
            samples.append(s)
    pos = sum(1 for s in samples for t in s["tags"] if t != "O")
    total = sum(len(s["tags"]) for s in samples)
    print(json.dumps({"documents": len(samples), "lines": total, "positive_markers": pos}, ensure_ascii=False))
    print(f"[+] Loaded {len(samples)} valid document samples with {total} lines and {pos} positive tags.")
    return samples


def split_by_file(samples: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    items = list(samples)
    rng.shuffle(items)
    n = len(items)
    a = max(1, int(0.8 * n)) if n >= 3 else max(1, n - 2)
    b = max(a + 1, int(0.9 * n)) if n >= 3 else max(a, n - 1)
    train = items[:a]
    val = items[a:b]
    test = items[b:]
    if not val and train:
        val = train[-1:]
        train = train[:-1] or val
    if not test and train:
        test = train[-1:]
    return train, val, test


def build_vocab(samples: list[dict[str, Any]], min_freq: int, max_vocab_size: int) -> tuple[dict[str, int], dict[str, int]]:
    wc = Counter()
    cc = Counter()
    for s in samples:
        for line_words in s["words"]:
            for w in line_words:
                wc[w] += 1
                wc[w.lower()] += 1
                for ch in w:
                    cc[ch] += 1
        for line in s["lines"]:
            for ch in line:
                cc[ch] += 1
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in wc.most_common():
        if max_vocab_size > 0 and len(word2idx) >= max_vocab_size:
            break
        if c >= min_freq and w not in word2idx:
            word2idx[w] = len(word2idx)
    for ch, c in cc.most_common():
        if c >= 1 and ch not in char2idx:
            char2idx[ch] = len(char2idx)
    return word2idx, char2idx


class ChunkDataset(Dataset):
    def __init__(self, samples, word2idx, char2idx, tag2idx, max_line_words, max_line_chars):
        self.samples = samples
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.max_line_words = max_line_words
        self.max_line_chars = max_line_chars

    def __len__(self):
        return len(self.samples)

    def wid(self, w: str) -> int:
        return self.word2idx.get(w, self.word2idx.get(w.lower(), self.word2idx["<UNK>"]))

    def __getitem__(self, idx):
        s = self.samples[idx]
        line_word_ids = []
        line_char_ids = []
        for line, words in zip(s["lines"], s["words"]):
            wids = [self.wid(w) for w in words[:self.max_line_words]]
            wids += [0] * (self.max_line_words - len(wids))
            cids = [self.char2idx.get(ch, self.char2idx["<UNK>"]) for ch in line[:self.max_line_chars]]
            cids += [0] * (self.max_line_chars - len(cids))
            line_word_ids.append(wids)
            line_char_ids.append(cids)
        tags = [self.tag2idx[t] for t in s["tags"]]
        return (
            torch.tensor(line_word_ids, dtype=torch.long),
            torch.tensor(line_char_ids, dtype=torch.long),
            torch.tensor(s["features"], dtype=torch.float),
            torch.tensor(tags, dtype=torch.long),
            len(s["lines"]),
            s["file_name"],
            s["lines"]
        )


def collate_fn(batch):
    word_ids, char_ids, feats, tags, lengths, file_names, lines = zip(*batch)
    word_pad = pad_sequence(word_ids, batch_first=True, padding_value=0)
    char_pad = pad_sequence(char_ids, batch_first=True, padding_value=0)
    feat_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
    tag_pad = pad_sequence(tags, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(word_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return word_pad, char_pad, feat_pad, tag_pad, lengths, mask, file_names, lines


class LineCharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, filters, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.conv3 = nn.Conv1d(char_emb_dim, filters, 3, padding=1)
        self.conv5 = nn.Conv1d(char_emb_dim, filters, 5, padding=2)
        self.act = nn.ReLU()

    def forward(self, char_ids):
        b, l, c = char_ids.shape
        x = char_ids.reshape(b * l, c)
        x = self.emb(x).permute(0, 2, 1)
        y3 = torch.max(self.act(self.conv3(x)), dim=-1)[0]
        y5 = torch.max(self.act(self.conv5(x)), dim=-1)[0]
        return torch.cat([y3, y5], dim=-1).reshape(b, l, -1)


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
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, emissions.size(1)):
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            score = score + (emit + trans) * mask[:, i]
        last_idx = mask.long().sum(1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def log_partition(self, emissions, mask):
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, emissions.size(1)):
            ns = torch.logsumexp(score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1), dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), ns, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        mask = mask.bool()
        return (self.log_partition(emissions, mask) - self.forward_score(emissions, tags, mask)).mean()

    def decode(self, emissions, mask):
        mask = mask.bool()
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, emissions.size(1)):
            ns = score.unsqueeze(2) + self.transitions + emissions[:, i].unsqueeze(1)
            best_score, best_path = ns.max(1)
            score = torch.where(mask[:, i].unsqueeze(1), best_score, score)
            history.append(best_path)
        score = score + self.end_transitions
        best_last = score.argmax(1)
        paths = []
        lengths = mask.long().sum(1)
        for bi in range(emissions.size(0)):
            seq_len = lengths[bi].item()
            last = best_last[bi].item()
            path = [last]
            for hist in reversed(history[:seq_len - 1]):
                last = hist[bi][last].item()
                path.append(last)
            path.reverse()
            paths.append(path)
        return paths


class ChunkMarkerModel(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, feature_dim, tag_size, word_emb_dim, char_emb_dim, char_filters, feat_dim, hidden_dim, dropout, pad_idx):
        super().__init__()
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=pad_idx)
        self.char_cnn = LineCharCNN(char_vocab_size, char_emb_dim, char_filters, pad_idx)
        self.feat_proj = nn.Sequential(nn.Linear(feature_dim, feat_dim), nn.ReLU())
        self.dropout_in = nn.Dropout(dropout)
        self.rnn = nn.GRU(word_emb_dim + char_filters * 2 + feat_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout_out = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
        self.crf = LinearCRF(tag_size, pad_idx)

    def line_word_repr(self, word_ids):
        emb = self.word_emb(word_ids)
        mask = (word_ids != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(2)
        denom = mask.sum(2).clamp_min(1.0)
        return summed / denom

    def emissions(self, word_ids, char_ids, feats, lengths):
        w = self.line_word_repr(word_ids)
        c = self.char_cnn(char_ids)
        f = self.feat_proj(feats)
        x = self.dropout_in(torch.cat([w, c, f], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=word_ids.size(1))
        out = self.dropout_out(out)
        return self.fc(out)

    def loss(self, word_ids, char_ids, feats, tags, lengths, mask, aux_weight=0.15):
        emissions = self.emissions(word_ids, char_ids, feats, lengths)
        crf_loss = self.crf.neg_log_likelihood(emissions, tags, mask)
        flat_em = emissions.reshape(-1, emissions.size(-1))
        flat_tags = tags.reshape(-1)
        flat_mask = mask.reshape(-1)
        weights = torch.ones(emissions.size(-1), device=emissions.device)
        for i in range(2, emissions.size(-1)):
            weights[i] = 8.0
        ce = nn.functional.cross_entropy(flat_em[flat_mask], flat_tags[flat_mask], weight=weights, ignore_index=0)
        return crf_loss + aux_weight * ce

    def decode(self, word_ids, char_ids, feats, lengths, mask):
        return self.crf.decode(self.emissions(word_ids, char_ids, feats, lengths), mask)


def compute_metrics(true_list, pred_list, idx2tag):
    stats = {tag: {"tp": 0, "fp": 0, "fn": 0} for tag in MARKER_ORDER}
    for true_tags, pred_tags in zip(true_list, pred_list):
        for tag in MARKER_ORDER:
            t = {i for i, x in enumerate(true_tags) if x == tag}
            p = {i for i, x in enumerate(pred_tags) if x == tag}
            stats[tag]["tp"] += len(t & p)
            stats[tag]["fp"] += len(p - t)
            stats[tag]["fn"] += len(t - p)
    total_tp = total_fp = total_fn = 0
    out = {}
    for tag in MARKER_ORDER:
        tp, fp, fn = stats[tag]["tp"], stats[tag]["fp"], stats[tag]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[tag] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out["micro"] = {"precision": p, "recall": r, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}
    return out


def evaluate(model, loader, idx2tag, device, name):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for word_ids, char_ids, feats, tags, lengths, mask, _, _ in tqdm(loader, desc=f"Evaluating {name}"):
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            feats = feats.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            loss = model.loss(word_ids, char_ids, feats, tags, lengths, mask, 0.0)
            losses.append(loss.item())
            paths = model.decode(word_ids, char_ids, feats, lengths, mask)
            for i, path in enumerate(paths):
                l = int(lengths[i].item())
                pred = [idx2tag[x] for x in path[:l]]
                true = [idx2tag[int(x)] for x in tags[i, :l].detach().cpu().tolist()]
                all_pred.append(pred)
                all_true.append(true)
    metrics = compute_metrics(all_true, all_pred, idx2tag)
    avg_loss = sum(losses) / max(1, len(losses))
    print(json.dumps({"split": name, "loss": avg_loss, "micro": metrics["micro"], "by_tag": {k: metrics[k] for k in MARKER_ORDER}}, ensure_ascii=False))
    print(f"[{name}] Loss: {avg_loss:.4f} | Micro Precision: {metrics['micro']['precision']:.4f} | Recall: {metrics['micro']['recall']:.4f} | F1: {metrics['micro']['f1']:.4f}")
    return avg_loss, metrics


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_bundle(output_dir, model, config, word2idx, char2idx, tag2idx):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "word2idx.json", word2idx)
    save_json(output_dir / "char2idx.json", char2idx)
    save_json(output_dir / "tag2idx.json", tag2idx)


def train(args):
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = load_samples(Path(args.markdown_folder), args.labels_jsonl, args.use_all_markdown)
    if not samples:
        raise ValueError("no samples")
    train_samples, val_samples, test_samples = split_by_file(samples, args.seed)
    print(f"[*] Split data: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test docs.")
    
    print("[*] Building vocabularies...")
    word2idx, char2idx = build_vocab(train_samples, args.min_freq, args.max_vocab_size)
    print(f"[+] Word vocabulary size: {len(word2idx)} (min_freq={args.min_freq})")
    print(f"[+] Char vocabulary size: {len(char2idx)}")
    
    tag2idx = {tag: i for i, tag in enumerate(TAG_LIST)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    
    print("[*] Preparing datasets and dataloaders...")
    train_ds = ChunkDataset(train_samples, word2idx, char2idx, tag2idx, args.max_line_words, args.max_line_chars)
    val_ds = ChunkDataset(val_samples, word2idx, char2idx, tag2idx, args.max_line_words, args.max_line_chars)
    test_ds = ChunkDataset(test_samples, word2idx, char2idx, tag2idx, args.max_line_words, args.max_line_chars)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Using device: {device}")
    
    print("[*] Initializing model...")
    model = ChunkMarkerModel(len(word2idx), len(char2idx), len(FEATURE_NAMES), len(TAG_LIST), args.word_emb_dim, args.char_emb_dim, args.char_filters, args.feature_proj_dim, args.hidden_dim, args.dropout, 0).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"train_docs": len(train_samples), "val_docs": len(val_samples), "test_docs": len(test_samples), "params": params, "device": str(device)}, ensure_ascii=False))
    print(f"[+] Model initialized with {params:,} parameters.")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    output_dir = Path(args.output_dir)
    config = {
        "word_emb_dim": args.word_emb_dim,
        "char_emb_dim": args.char_emb_dim,
        "char_filters": args.char_filters,
        "feature_proj_dim": args.feature_proj_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_line_words": args.max_line_words,
        "max_line_chars": args.max_line_chars,
        "feature_names": FEATURE_NAMES,
        "marker_order": MARKER_ORDER,
        "max_vocab_size": args.max_vocab_size,
        "rnn_type": "gru"
    }
    best = -1.0
    bad = 0
    
    print(f"[*] Starting training for {args.epochs} epochs with early stopping patience {args.early_stop}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for word_ids, char_ids, feats, tags, lengths, mask, _, _ in pbar:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            feats = feats.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            loss = model.loss(word_ids, char_ids, feats, tags, lengths, mask, args.aux_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = sum(losses) / max(1, len(losses))
        print(json.dumps({"epoch": epoch, "train_loss": avg_loss}, ensure_ascii=False))
        print(f"[Epoch {epoch}] Average training loss: {avg_loss:.4f}")
        
        _, val_metrics = evaluate(model, val_loader, idx2tag, device, "val")
        val_f1 = val_metrics["micro"]["f1"]
        scheduler.step(val_f1)
        if val_f1 > best:
            best = val_f1
            bad = 0
            save_bundle(output_dir, model, config, word2idx, char2idx, tag2idx)
            print(json.dumps({"saved": str(output_dir), "best_val_f1": best}, ensure_ascii=False))
            print(f"[Epoch {epoch}] New best validation F1: {best:.4f}. Saved model bundle to {output_dir}")
        else:
            bad += 1
            print(f"[Epoch {epoch}] F1 did not improve. Bad epochs: {bad}/{args.early_stop}")
            if args.early_stop > 0 and bad >= args.early_stop:
                print("[*] Early stopping triggered.")
                break
    print("[*] Loading best saved model weights for testing...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    evaluate(model, test_loader, idx2tag, device, "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown_folder", required=True)
    parser.add_argument("--labels_jsonl", nargs="+", required=True)
    parser.add_argument("--output_dir", default="artifacts/chunk_marker_ner")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--word_emb_dim", type=int, default=40)
    parser.add_argument("--char_emb_dim", type=int, default=12)
    parser.add_argument("--char_filters", type=int, default=16)
    parser.add_argument("--feature_proj_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=40)
    parser.add_argument("--max_line_words", type=int, default=28)
    parser.add_argument("--max_line_chars", type=int, default=160)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=8000)
    parser.add_argument("--aux_weight", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_all_markdown", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
