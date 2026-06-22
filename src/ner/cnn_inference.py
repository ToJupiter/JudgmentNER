import json
import re
import argparse
from typing import List, Optional

import polars as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text)


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int = 25,
                 num_filters: int = 30, kernel_size: int = 3, pad_idx: int = 0):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.activation = nn.Tanh()

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        B, S, W = char_ids.shape
        char_ids = char_ids.view(-1, W)
        char_emb = self.char_embed(char_ids)
        char_emb = char_emb.permute(0, 2, 1)
        conv_out = self.conv(char_emb)
        char_feat = torch.max(self.activation(conv_out), dim=-1)[0]
        return char_feat.view(B, S, -1)


class CNNBiLSTM_NER(nn.Module):
    def __init__(self, word_vocab_size: int, char_vocab_size: int, tag_size: int,
                 word_emb_dim: int = 64, char_emb_dim: int = 25, num_filters: int = 30,
                 lstm_hidden_dim: int = 64, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=pad_idx)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, num_filters, pad_idx=pad_idx)
        self.input_dim = word_emb_dim + num_filters
        self.dropout_emb = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, tag_size)

    def forward(self, word_ids: torch.Tensor, char_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        word_emb = self.word_embed(word_ids)
        char_feat = self.char_cnn(char_ids)
        combined = torch.cat([word_emb, char_feat], dim=-1)
        combined = self.dropout_emb(combined)

        packed_input = pack_padded_sequence(combined, lengths.cpu(),
                                            batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout_lstm(lstm_out)
        return self.fc(lstm_out)


def load_model_and_vocab(model_path: str, word2idx_path: str, char2idx_path: str):
    with open(word2idx_path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    with open(char2idx_path, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    word_vocab_size = len(word2idx)
    char_vocab_size = len(char2idx)
    tag_size = 6
    model = CNNBiLSTM_NER(word_vocab_size, char_vocab_size, tag_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, word2idx, char2idx


def infer_snippet(snippet: str, model: nn.Module, word2idx: dict, char2idx: dict,
                  max_word_len: int = 20) -> (List[str], List[str]):
    tokens = tokenize(snippet)
    if not tokens:
        return [], []

    word_ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    char_ids = []
    for w in tokens:
        char_seq = [char2idx.get(c, char2idx["<UNK>"]) for c in w[:max_word_len]]
        char_seq += [char2idx["<PAD>"]] * (max_word_len - len(char_seq))
        char_ids.append(char_seq)

    word_t = torch.tensor([word_ids], dtype=torch.long)
    char_t = torch.tensor([char_ids], dtype=torch.long)
    lengths = torch.tensor([len(tokens)])

    with torch.no_grad():
        logits = model(word_t, char_t, lengths)
        preds = torch.argmax(logits, dim=-1)
        pred_ids = preds[0].tolist()[:len(tokens)]

    idx2tag = {0: "<PAD>", 1: "O", 2: "B-LAW", 3: "I-LAW", 4: "B-ART", 5: "I-ART"}
    tags = [idx2tag[p] for p in pred_ids]
    return tokens, tags


def extract_citations(tokens: List[str], tags: List[str]) -> List[dict]:
    laws = []
    current_law = None
    idx = 0
    while idx < len(tags):
        tag = tags[idx]
        if tag == "B-LAW":
            if current_law is not None:
                laws.append(current_law)
            name_parts = [tokens[idx]]
            idx += 1
            while idx < len(tags) and tags[idx] == "I-LAW":
                name_parts.append(tokens[idx])
                idx += 1
            current_law = {"law_name": " ".join(name_parts), "articles": []}
        else:
            if tag == "I-ART" and current_law is not None:
                current_law["articles"].append(tokens[idx])
            idx += 1
    if current_law is not None:
        laws.append(current_law)
    return laws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', required=True, help='Path to ocr_cache.parquet')
    parser.add_argument('--model', required=True, help='Path to best_model.pt')
    parser.add_argument('--word2idx', required=True, help='Path to word2idx.json')
    parser.add_argument('--char2idx', required=True, help='Path to char2idx.json')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    parser.add_argument('--regex', default=r'(?:các\s+)?(?:Điều|khoản|điểm)\s+[\d,\s]+(?:và\s+[\d,\s]+)?[^.]*\.',
                        help='Regex pattern for citation extraction')
    args = parser.parse_args()

    model, word2idx, char2idx = load_model_and_vocab(args.model, args.word2idx, args.char2idx)

    df = pl.read_parquet(args.parquet)
    output_data = []

    for row in df.rows(named=True):
        pdf_path = row['pdf_path']
        ocr_text = row['ocr_text']
        if not ocr_text:
            continue

        matches = re.findall(args.regex, ocr_text, re.IGNORECASE)
        citations_for_pdf = []

        for match in matches:
            tokens, tags = infer_snippet(match, model, word2idx, char2idx)
            if not tokens:
                continue
            laws = extract_citations(tokens, tags)
            for law in laws:
                if law["articles"]:
                    citations_for_pdf.append({
                        "law_name": law["law_name"],
                        "articles": law["articles"]
                    })

        if citations_for_pdf:
            output_data.append({
                "pdf_path": pdf_path,
                "citations": citations_for_pdf
            })

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()