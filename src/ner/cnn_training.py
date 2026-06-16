#!/usr/bin/env python3
import json
import re
import random
import argparse
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


def normalize_law_name(name: str) -> str:
    return name.strip().rstrip('.,;: ').lower()


def find_substring_span(text: str, substring: str, ignore_case: bool = True) -> Optional[Tuple[int, int]]:
    if ignore_case:
        idx = text.lower().find(substring.lower())
    else:
        idx = text.find(substring)
    if idx == -1:
        return None
    return idx, idx + len(substring)


def char_to_token_spans(tokens: List[str], text: str) -> List[Tuple[int, int]]:
    spans = []
    pos = 0
    for t in tokens:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        start = pos
        end = pos + len(t)
        spans.append((start, end))
        pos = end
    return spans


def build_vocab(sentences: List[List[str]], min_freq: int = 2):
    word_counter = Counter()
    char_set = set()
    for sent in sentences:
        for w in sent:
            word_counter[w] += 1
            char_set.update(w)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in word_counter.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    for ch in sorted(char_set):
        char2idx[ch] = len(char2idx)
    print(f"[VOCAB] Word vocab size: {len(word2idx)}, Char vocab size: {len(char2idx)}")
    return word2idx, char2idx


def load_law_database(csv_path: str):
    import csv
    titles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                titles.append(row[0].strip().strip('"'))
    print(f"[LAW DB] Loaded {len(titles)} titles")
    return titles


def match_law_and_articles(entry: Dict, tokens: List[str], token_spans: List[Tuple[int, int]], text: str):
    tags = ["O"] * len(tokens)
    errors = []

    for cit in entry["citations"]:
        law_raw = cit["law"]
        articles = cit["articles"]

        span_char = find_substring_span(text, law_raw, ignore_case=False)
        if span_char is None:
            span_char = find_substring_span(text, law_raw.rstrip('.,;: '), ignore_case=False)
        if span_char is None:
            span_char = find_substring_span(text, law_raw, ignore_case=True)
        if span_char is None:
            span_char = find_substring_span(text, law_raw.rstrip('.,;: '), ignore_case=True)
        if span_char is None:
            errors.append(f"Law span not found: {law_raw}")
            continue

        start_c, end_c = span_char
        law_indices = [i for i, (s, e) in enumerate(token_spans) if e > start_c and s < end_c]
        if not law_indices:
            errors.append(f"Law token indices empty: {law_raw}")
            continue

        for idx_i, tidx in enumerate(law_indices):
            tags[tidx] = "B-LAW" if idx_i == 0 else "I-LAW"

        for art in articles:
            art_token_idx = None
            for i, t in enumerate(tokens):
                clean_t = t.strip('.,;:()[]{}')
                if clean_t == art:
                    art_token_idx = i
                    break
            if art_token_idx is None:
                for i, t in enumerate(tokens):
                    if t == art:
                        art_token_idx = i
                        break
            if art_token_idx is None:
                errors.append(f"Article number token not found: {art}")
                continue

            dieu_found = False
            for j in range(art_token_idx - 1, -1, -1):
                if tokens[j].lower() == "điều" and tags[j] == "O":
                    tags[j] = "B-ART"
                    dieu_found = True
                    break
            if dieu_found:
                tags[art_token_idx] = "I-ART"
            else:
                tags[art_token_idx] = "B-ART"

    return tags, errors


def prepare_data(jsonl_files: List[str], law_csv: Optional[str] = None):
    entries = []
    file_stats = {}

    for fname in jsonl_files:
        count = 0
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except:
                    continue
                if entry.get("citations") and len(entry["citations"]) > 0:
                    entries.append(entry)
                    count += 1
        file_stats[fname] = count
        print(f"[DATA] File {fname}: {count} valid entries with citations")

    total_before_dedup = len(entries)
    print(f"[DATA] Total entries before dedup: {total_before_dedup}")

    def key_fn(e):
        return (e.get("file_name", ""), e.get("entry_index", -1), e.get("law_index", -1))
    unique_entries = {}
    duplicates = 0
    for entry in entries:
        key = key_fn(entry)
        if key not in unique_entries:
            unique_entries[key] = entry
        else:
            duplicates += 1
    print(f"[DATA] Dedup by (file_name, entry_index, law_index): removed {duplicates} duplicates, {len(unique_entries)} unique")

    samples = []
    fail_reasons = Counter()
    success_count = 0

    for key, entry in unique_entries.items():
        inp = entry["input"]
        tokens = tokenize(inp)
        token_spans = char_to_token_spans(tokens, inp)
        tags, errors = match_law_and_articles(entry, tokens, token_spans, inp)
        if errors:
            for err in errors:
                fail_reasons[err] += 1
            continue
        samples.append((tokens, tags, inp))
        success_count += 1

    print(f"[DATA] Successfully tagged: {success_count}")
    if fail_reasons:
        print(f"[DATA] Top failure reasons: {fail_reasons.most_common(15)}")
    if samples:
        print("[DATA] First 3 tagged samples:")
        for tokens, tags, inp in samples[:3]:
            print(f"  Input: {inp}")
            print(f"  Tokens: {tokens}")
            print(f"  Tags:   {tags}")
    return samples


class NERDataset(Dataset):
    def __init__(self, samples, word2idx, char2idx, tag2idx, max_word_len=20):
        self.samples = samples
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, tags, _ = self.samples[idx]
        word_ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        char_ids = []
        for w in tokens:
            char_seq = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in w[:self.max_word_len]]
            char_seq += [self.char2idx["<PAD>"]] * (self.max_word_len - len(char_seq))
            char_ids.append(char_seq)
        tag_ids = [self.tag2idx[t] for t in tags]
        return torch.tensor(word_ids), torch.tensor(char_ids), torch.tensor(tag_ids), len(tokens)


def collate_fn(batch):
    word_ids, char_ids, tag_ids, lengths = zip(*batch)
    word_pad = pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_pad = pad_sequence(tag_ids, batch_first=True, padding_value=0)
    char_pad = pad_sequence(char_ids, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    return word_pad, char_pad, tag_pad, lengths


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=25, num_filters=30, kernel_size=3, pad_idx=0):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.activation = nn.Tanh()

    def forward(self, char_ids):
        B, S, W = char_ids.shape
        char_ids = char_ids.view(-1, W)
        char_emb = self.char_embed(char_ids)
        char_emb = char_emb.permute(0, 2, 1)
        conv_out = self.conv(char_emb)
        char_feat = torch.max(self.activation(conv_out), dim=-1)[0]
        return char_feat.view(B, S, -1)


class CNNBiLSTM_NER(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, tag_size,
                 word_emb_dim=64, char_emb_dim=25, num_filters=30,
                 lstm_hidden_dim=64, dropout=0.3, pad_idx=0):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=pad_idx)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, num_filters, pad_idx=pad_idx)
        self.input_dim = word_emb_dim + num_filters
        self.dropout_emb = nn.Dropout(dropout)

        self.lstm = nn.LSTM(self.input_dim, lstm_hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, tag_size)

    def forward(self, word_ids, char_ids, lengths):
        word_emb = self.word_embed(word_ids)
        char_feat = self.char_cnn(char_ids)
        combined = torch.cat([word_emb, char_feat], dim=-1)
        combined = self.dropout_emb(combined)

        packed_input = pack_padded_sequence(combined, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout_lstm(lstm_out)
        logits = self.fc(lstm_out)
        return logits


def compute_metrics(true_tags_list, pred_tags_list):
    def get_spans(tags, target_tag):
        spans = set()
        start = None
        for i, t in enumerate(tags):
            if t == f"B-{target_tag}":
                start = i
            elif t == f"I-{target_tag}":
                continue
            else:
                if start is not None:
                    spans.add((start, i - 1))
                    start = None
        if start is not None:
            spans.add((start, len(tags) - 1))
        return spans

    law_tp = law_fp = law_fn = 0
    art_tp = art_fp = art_fn = 0
    for t_tags, p_tags in zip(true_tags_list, pred_tags_list):
        t_law = get_spans(t_tags, "LAW")
        p_law = get_spans(p_tags, "LAW")
        t_art = get_spans(t_tags, "ART")
        p_art = get_spans(p_tags, "ART")
        law_tp += len(t_law & p_law)
        law_fp += len(p_law - t_law)
        law_fn += len(t_law - p_law)
        art_tp += len(t_art & p_art)
        art_fp += len(p_art - t_art)
        art_fn += len(t_art - p_art)

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    law_prec, law_rec, law_f1 = prf(law_tp, law_fp, law_fn)
    art_prec, art_rec, art_f1 = prf(art_tp, art_fp, art_fn)
    return {"law_prec": law_prec, "law_rec": law_rec, "law_f1": law_f1,
            "art_prec": art_prec, "art_rec": art_rec, "art_f1": art_f1,
            "law_tp": law_tp, "law_fp": law_fp, "law_fn": law_fn,
            "art_tp": art_tp, "art_fp": art_fp, "art_fn": art_fn}


def evaluate(model, dataloader, idx2tag, device, detailed=True):
    model.eval()
    total_loss = 0.0
    all_true_tags, all_pred_tags = [], []
    token_correct, token_total = 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for word_ids, char_ids, tag_ids, lengths in dataloader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device)
            logits = model(word_ids, char_ids, lengths)
            loss = criterion(logits.view(-1, logits.shape[-1]), tag_ids.view(-1))
            total_loss += loss.item() * word_ids.size(0)
            preds = torch.argmax(logits, dim=-1)
            mask = (tag_ids != 0)
            token_correct += (preds == tag_ids).masked_select(mask).sum().item()
            token_total += mask.sum().item()
            preds_cpu = preds.cpu().numpy()
            tags_cpu = tag_ids.cpu().numpy()
            for i in range(len(lengths)):
                l = lengths[i]
                p_tags = [idx2tag[p] for p in preds_cpu[i][:l]]
                t_tags = [idx2tag[t] for t in tags_cpu[i][:l]]
                all_true_tags.append(t_tags)
                all_pred_tags.append(p_tags)
    avg_loss = total_loss / len(dataloader.dataset)
    token_acc = token_correct / token_total if token_total > 0 else 0.0
    metrics = compute_metrics(all_true_tags, all_pred_tags)
    if detailed:
        print(f"  Token acc: {token_acc:.4f}")
        print(f"  LAW - P: {metrics['law_prec']:.4f}, R: {metrics['law_rec']:.4f}, F1: {metrics['law_f1']:.4f} (tp:{metrics['law_tp']}, fp:{metrics['law_fp']}, fn:{metrics['law_fn']})")
        print(f"  ART - P: {metrics['art_prec']:.4f}, R: {metrics['art_rec']:.4f}, F1: {metrics['art_f1']:.4f} (tp:{metrics['art_tp']}, fp:{metrics['art_fp']}, fn:{metrics['art_fn']})")
    return avg_loss, token_acc, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_files', nargs='+', required=True)
    parser.add_argument('--law_csv', default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_output', default=None, help='Output file for tokenisation samples')
    parser.add_argument('--sample_count', type=int, default=200)
    args = parser.parse_args()

    print(f"[CONFIG] Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Seed: {args.seed}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    law_titles = None
    if args.law_csv:
        law_titles = load_law_database(args.law_csv)

    samples = prepare_data(args.jsonl_files, args.law_csv)
    if not samples:
        raise ValueError("No valid samples found. Check your data.")

    if args.sample_output and args.sample_count > 0:
        count = min(args.sample_count, len(samples))
        with open(args.sample_output, 'w', encoding='utf-8') as f:
            for i in range(count):
                tokens, tags, inp = samples[i]
                f.write(f"Input: {inp}\n")
                f.write(f"Tokens: {' '.join(tokens)}\n")
                f.write(f"Tags:   {' '.join(tags)}\n\n")
        print(f"[SAMPLE] Wrote {count} tokenisation samples to {args.sample_output}")

    random.shuffle(samples)
    n = len(samples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    print(f"[SPLIT] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_sents = [tok for tok, _, _ in train_samples]
    word2idx, char2idx = build_vocab(train_sents, min_freq=2)
    tag2idx = {"O": 0, "B-LAW": 1, "I-LAW": 2, "B-ART": 3, "I-ART": 4}
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_ds = NERDataset(train_samples, word2idx, char2idx, tag2idx)
    val_ds = NERDataset(val_samples, word2idx, char2idx, tag2idx)
    test_ds = NERDataset(test_samples, word2idx, char2idx, tag2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNNBiLSTM_NER(len(word2idx), len(char2idx), len(tag2idx))
    model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] CNNBiLSTM - Total params: {total_p:,}, Trainable: {trainable_p:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        for word_ids, char_ids, tag_ids, lengths in train_loader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device)
            optimizer.zero_grad()
            logits = model(word_ids, char_ids, lengths)
            loss = criterion(logits.view(-1, logits.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * word_ids.size(0)
            preds = torch.argmax(logits, dim=-1)
            mask = (tag_ids != 0)
            train_correct += (preds == tag_ids).masked_select(mask).sum().item()
            train_total += mask.sum().item()
        train_loss = total_loss / len(train_ds)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        print(f"[EPOCH {epoch:2d}] Train loss: {train_loss:.4f}, token acc: {train_acc:.4f}")

        val_loss, val_acc, val_metrics = evaluate(model, val_loader, idx2tag, device, detailed=True)
        val_f1 = (val_metrics['law_f1'] + val_metrics['art_f1']) / 2
        print(f"  Val loss: {val_loss:.4f}, avg F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            print("  ** Saved best model **")

    print("\n" + "=" * 50)
    print("[TEST] Final evaluation:")
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc, test_metrics = evaluate(model, test_loader, idx2tag, device, detailed=True)

    if args.law_csv and law_titles:
        print("\n[POST-PROCESS] Law title matching examples:")
        from difflib import get_close_matches

        def match_law(pred_name, title_list, cutoff=0.8):
            if not pred_name:
                return ""
            matches = get_close_matches(pred_name, title_list, n=1, cutoff=cutoff)
            return matches[0] if matches else pred_name

        model.eval()
        demo_n = min(5, len(test_samples))
        with torch.no_grad():
            for i in range(demo_n):
                tokens, true_tags, inp = test_samples[i]
                word_ids, char_ids, tag_ids, lengths = test_ds[i]
                word_ids = word_ids.unsqueeze(0).to(device)
                char_ids = char_ids.unsqueeze(0).to(device)
                logits = model(word_ids, char_ids, torch.tensor([lengths]))
                pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_tags = [idx2tag[p] for p in pred_ids[:lengths]]
                law_tokens = [t for t, ptag in zip(tokens, pred_tags) if ptag in ("B-LAW", "I-LAW")]
                raw_law = " ".join(law_tokens)
                matched = match_law(raw_law, law_titles)
                print(f"  Input: {inp}")
                print(f"    True: {true_tags}")
                print(f"    Pred: {pred_tags}")
                print(f"    Law prediction: '{raw_law}' -> matched: '{matched}'")
                print()

    if device.type == 'cuda':
        print(f"[DEVICE] Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    main()