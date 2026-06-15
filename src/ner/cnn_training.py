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
from torch.nn.utils.rnn import pad_sequence


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


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
    print(f"[DATA] Total entries with citations before dedup: {total_before_dedup}")

    key_fn = lambda e: (e.get("file_name", ""), e.get("entry_index", -1), e.get("law_index", -1))
    unique_entries = {}
    duplicates = 0
    for entry in entries:
        key = key_fn(entry)
        if key not in unique_entries:
            unique_entries[key] = entry
        else:
            duplicates += 1
    print(f"[DATA] Dedup based on (file_name, entry_index, law_index): {duplicates} duplicates removed, {len(unique_entries)} unique entries")

    samples = []
    fail_reasons = Counter()
    success_count = 0

    for key, entry in unique_entries.items():
        inp = entry["input"]
        tokens = tokenize(inp)
        token_spans = char_to_token_spans(tokens, inp)
        tags = ["O"] * len(tokens)

        success = True
        for cit in entry["citations"]:
            law_str = cit["law"]
            articles = cit["articles"]

            law_span_char = None
            law_span_char = find_substring_span(inp, law_str, ignore_case=False)
            if law_span_char is None:
                law_clean = law_str.rstrip('.,;: ')
                law_span_char = find_substring_span(inp, law_clean, ignore_case=False)
            if law_span_char is None:
                law_span_char = find_substring_span(inp, law_str, ignore_case=True)
                if law_span_char is None:
                    law_span_char = find_substring_span(inp, law_str.rstrip('.,;: '), ignore_case=True)
            if law_span_char is None:
                success = False
                fail_reasons["law_span_not_found"] += 1
                break

            start_c, end_c = law_span_char
            law_token_indices = []
            for i, (s, e) in enumerate(token_spans):
                if e > start_c and s < end_c:
                    law_token_indices.append(i)
            if not law_token_indices:
                success = False
                fail_reasons["law_token_indices_empty"] += 1
                break

            for idx_i, tidx in enumerate(law_token_indices):
                if idx_i == 0:
                    tags[tidx] = "B-LAW"
                else:
                    tags[tidx] = "I-LAW"

            art_found_count = 0
            for art in articles:
                art_token_idx = None
                for i, t in enumerate(tokens):
                    clean_t = t.strip('.,;:')
                    if clean_t == art:
                        art_token_idx = i
                        break
                if art_token_idx is not None:
                    tags[art_token_idx] = "B-ART"
                    art_found_count += 1
            if art_found_count != len(articles):
                pass
        if not success:
            continue
        samples.append((tokens, tags))
        success_count += 1

    print(f"[DATA] Successfully tagged samples: {success_count}")
    if fail_reasons:
        print(f"[DATA] Failure reasons: {dict(fail_reasons)}")
    if len(samples) > 0:
        print(f"[DATA] Example tagged sentence (first 5):")
        for tokens, tags in samples[:5]:
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
        tokens, tags = self.samples[idx]
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


class CNN_NER(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, tag_size,
                 word_emb_dim=100, char_emb_dim=25, num_filters=30,
                 hidden_dim=64, dropout=0.3, pad_idx=0):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=pad_idx)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, num_filters)
        self.dropout_emb = nn.Dropout(dropout)
        input_dim = word_emb_dim + num_filters
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, word_ids, char_ids, lengths):
        word_emb = self.word_embed(word_ids)
        char_feat = self.char_cnn(char_ids)
        combined = torch.cat([word_emb, char_feat], dim=-1)
        combined = self.dropout_emb(combined)

        x = combined.permute(0, 2, 1)
        x = self.activation(self.conv1(x))
        x = self.dropout_cnn(x)
        x = self.activation(self.conv2(x))
        x = x.permute(0, 2, 1)
        logits = self.fc(x)
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
                    spans.add((start, i-1))
                    start = None
        if start is not None:
            spans.add((start, len(tags)-1))
        return spans

    law_tp = law_fp = law_fn = 0
    art_tp = art_fp = art_fn = 0
    total_true_law = 0
    total_true_art = 0
    total_pred_law = 0
    total_pred_art = 0

    for t_tags, p_tags in zip(true_tags_list, pred_tags_list):
        t_law = get_spans(t_tags, "LAW")
        p_law = get_spans(p_tags, "LAW")
        t_art = get_spans(t_tags, "ART")
        p_art = get_spans(p_tags, "ART")

        total_true_law += len(t_law)
        total_true_art += len(t_art)
        total_pred_law += len(p_law)
        total_pred_art += len(p_art)

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

    return {
        "law_precision": law_prec, "law_recall": law_rec, "law_f1": law_f1,
        "art_precision": art_prec, "art_recall": art_rec, "art_f1": art_f1,
        "law_true_count": total_true_law, "law_pred_count": total_pred_law,
        "art_true_count": total_true_art, "art_pred_count": total_pred_art,
    }


def evaluate(model, dataloader, idx2tag, device, detailed=True):
    model.eval()
    total_loss = 0.0
    all_true_tags = []
    all_pred_tags = []
    token_correct = 0
    token_total = 0
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
        print(f"  Token accuracy: {token_acc:.4f}")
        print(f"  LAW - Precision: {metrics['law_precision']:.4f}, Recall: {metrics['law_recall']:.4f}, F1: {metrics['law_f1']:.4f} | True: {metrics['law_true_count']}, Pred: {metrics['law_pred_count']}")
        print(f"  ART - Precision: {metrics['art_precision']:.4f}, Recall: {metrics['art_recall']:.4f}, F1: {metrics['art_f1']:.4f} | True: {metrics['art_true_count']}, Pred: {metrics['art_pred_count']}")
    return avg_loss, token_acc, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_files', nargs='+', required=True,
                        help='List of JSONL files')
    parser.add_argument('--law_csv', default=None,
                        help='Optional law title CSV for post-processing demo')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"[CONFIG] Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, Seed: {args.seed}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    samples = prepare_data(args.jsonl_files, args.law_csv)
    if len(samples) == 0:
        raise ValueError("No valid samples found. Check your data.")

    random.shuffle(samples)
    n = len(samples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    print(f"[SPLIT] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_sents = [tokens for tokens, _ in train_samples]
    word2idx, char2idx = build_vocab(train_sents, min_freq=2)

    tag2idx = {"O": 0, "B-LAW": 1, "I-LAW": 2, "B-ART": 3, "I-ART": 4}
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_ds = NERDataset(train_samples, word2idx, char2idx, tag2idx)
    val_ds = NERDataset(val_samples, word2idx, char2idx, tag2idx)
    test_ds = NERDataset(test_samples, word2idx, char2idx, tag2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNN_NER(len(word2idx), len(char2idx), len(tag2idx))
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        train_token_correct = 0
        train_token_total = 0
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
            train_token_correct += (preds == tag_ids).masked_select(mask).sum().item()
            train_token_total += mask.sum().item()

        avg_train_loss = total_loss / len(train_ds)
        train_token_acc = train_token_correct / train_token_total if train_token_total > 0 else 0.0

        print(f"[EPOCH {epoch:3d}] Train loss: {avg_train_loss:.4f}, Train token acc: {train_token_acc:.4f}")

        print("[VAL] Validation results:")
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, idx2tag, device, detailed=True)
        avg_val_f1 = (val_metrics['law_f1'] + val_metrics['art_f1']) / 2
        print(f"  Val loss: {val_loss:.4f}, avg F1: {avg_val_f1:.4f}")

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), "best_ner.pt")
            print(f"  ** New best model saved (avg F1: {best_val_f1:.4f}) **")

    print("\n" + "="*50)
    print("[TEST] Final test evaluation:")
    model.load_state_dict(torch.load("best_ner.pt"))
    test_loss, test_acc, test_metrics = evaluate(model, test_loader, idx2tag, device, detailed=True)
    print(f"  Test loss: {test_loss:.4f}, Token acc: {test_acc:.4f}")

    if args.law_csv:
        print("\n" + "="*50)
        print("[POST-PROCESS] Law title matching demo")
        import csv
        from difflib import get_close_matches
        law_titles = []
        with open(args.law_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    law_titles.append(row[0].strip())
        print(f"[POST-PROCESS] Loaded {len(law_titles)} law titles from CSV")

        def match_law(predicted_name, title_list, cutoff=0.8):
            matches = get_close_matches(predicted_name, title_list, n=1, cutoff=cutoff)
            return matches[0] if matches else predicted_name

        model.eval()
        demo_count = min(5, len(test_samples))
        print(f"[POST-PROCESS] Displaying {demo_count} examples:")
        with torch.no_grad():
            for i in range(demo_count):
                tokens, true_tags = test_samples[i]
                word_ids, char_ids, tag_ids, lengths = test_ds[i]
                word_ids = word_ids.unsqueeze(0).to(device)
                char_ids = char_ids.unsqueeze(0).to(device)
                logits = model(word_ids, char_ids, torch.tensor([lengths]))
                pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_tags = [idx2tag[p] for p in pred_ids[:lengths]]

                law_tokens = []
                for t, ptag in zip(tokens, pred_tags):
                    if ptag in ("B-LAW", "I-LAW"):
                        law_tokens.append(t)
                raw_law = " ".join(law_tokens)
                matched = match_law(raw_law, law_titles) if raw_law else ""
                print(f"  Input: {' '.join(tokens)}")
                print(f"    True tags: {true_tags}")
                print(f"    Pred tags: {pred_tags}")
                print(f"    Pred law: '{raw_law}' -> matched: '{matched}'")
                print()

    if device.type == 'cuda':
        print(f"\n[DEVICE] Peak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()