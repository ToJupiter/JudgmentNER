import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cnn_training_chunk import (
    MARKER_ORDER,
    TAG_LIST,
    COMPILED_PATTERNS,
    FEATURE_NAMES,
    ChunkMarkerModel,
    compact_nonempty_lines,
    contextualize_features,
    extract_line_features,
    load_json,
    normalize_line,
    split_lines,
    tokenize_words
)


def load_bundle(model_dir: Path, device):
    config = load_json(model_dir / "config.json")
    word2idx = load_json(model_dir / "word2idx.json")
    char2idx = load_json(model_dir / "char2idx.json")
    tag2idx = load_json(model_dir / "tag2idx.json")
    idx2tag = {int(v): k for k, v in tag2idx.items()}
    model = ChunkMarkerModel(
        len(word2idx),
        len(char2idx),
        len(FEATURE_NAMES),
        len(tag2idx),
        config["word_emb_dim"],
        config["char_emb_dim"],
        config["char_filters"],
        config["feature_proj_dim"],
        config["hidden_dim"],
        config["dropout"],
        0
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, config, word2idx, char2idx, tag2idx, idx2tag


def word_id(word2idx: dict, w: str) -> int:
    return word2idx.get(w, word2idx.get(w.lower(), word2idx["<UNK>"]))


def encode_lines(lines, word2idx, char2idx, config, device):
    max_line_words = config["max_line_words"]
    max_line_chars = config["max_line_chars"]
    word_rows = []
    char_rows = []
    base_feats = []
    total = len(lines)
    for i, line in enumerate(lines):
        words = tokenize_words(line)
        wids = [word_id(word2idx, w) for w in words[:max_line_words]]
        wids += [0] * (max_line_words - len(wids))
        cids = [char2idx.get(ch, char2idx["<UNK>"]) for ch in line[:max_line_chars]]
        cids += [0] * (max_line_chars - len(cids))
        word_rows.append(wids)
        char_rows.append(cids)
        base_feats.append(extract_line_features(line, i, total))
    feats = contextualize_features(lines, base_feats)
    word_tensor = torch.tensor([word_rows], dtype=torch.long, device=device)
    char_tensor = torch.tensor([char_rows], dtype=torch.long, device=device)
    feat_tensor = torch.tensor([feats], dtype=torch.float, device=device)
    lengths = torch.tensor([len(lines)], dtype=torch.long, device=device)
    mask = torch.ones(1, len(lines), dtype=torch.bool, device=device)
    return word_tensor, char_tensor, feat_tensor, lengths, mask


def regex_hit(line: str, marker_type: str) -> bool:
    return any(p.search(line) for p in COMPILED_PATTERNS[marker_type])


def position_prior(marker_type: str, pos: float) -> float:
    if marker_type == "preamble":
        return 0.45 if pos <= 0.2 else -0.3
    if marker_type == "noi_dung_vu_an":
        return 0.25 if 0.02 <= pos <= 0.45 else -0.1
    if marker_type == "dai_dien_vks":
        return 0.15 if 0.15 <= pos <= 0.75 else -0.05
    if marker_type == "nhan_dinh":
        return 0.25 if 0.25 <= pos <= 0.88 else -0.15
    if marker_type == "quyet_dinh":
        return 0.45 if pos >= 0.45 else -0.3
    return 0.0


def fallback_regex(lines, marker_type, start, end):
    best = None
    for i in range(max(0, start), min(len(lines), end)):
        if regex_hit(lines[i], marker_type):
            score = 1.0 + position_prior(marker_type, i / max(1, len(lines) - 1))
            item = {"line_index": i, "marker_type": marker_type, "line": lines[i], "score": score, "source": "regex_fallback"}
            if best is None or item["score"] > best["score"]:
                best = item
    return best


def choose_markers(lines, decoded_tags, probs, tag2idx, threshold):
    candidates = {m: [] for m in MARKER_ORDER}
    n = len(lines)
    for i, line in enumerate(lines):
        pos = i / max(1, n - 1)
        for marker_type in MARKER_ORDER:
            idx = int(tag2idx[marker_type])
            score = float(probs[i][idx])
            if decoded_tags[i] == marker_type:
                score += 0.75
            if regex_hit(line, marker_type):
                score += 0.55
            score += position_prior(marker_type, pos)
            if score >= threshold or regex_hit(line, marker_type) or decoded_tags[i] == marker_type:
                candidates[marker_type].append({"line_index": i, "marker_type": marker_type, "line": line, "score": score, "source": "model_regex"})
    selected = {}
    warnings = []
    last = -1
    for idx, marker_type in enumerate(MARKER_ORDER):
        viable = [x for x in candidates[marker_type] if x["line_index"] > last]
        if marker_type == "preamble":
            early = [x for x in viable if x["line_index"] <= max(30, int(n * 0.18))]
            if early:
                viable = early
        if marker_type == "quyet_dinh":
            late = [x for x in viable if x["line_index"] >= int(n * 0.45)]
            if late:
                viable = late
        viable.sort(key=lambda x: (x["score"], -abs(x["line_index"] - last)), reverse=True)
        chosen = viable[0] if viable else None
        if chosen is None:
            next_end = n
            chosen = fallback_regex(lines, marker_type, last + 1, next_end)
            if chosen is not None:
                warnings.append(f"fallback_regex:{marker_type}")
        if chosen is not None and chosen["line_index"] > last:
            selected[marker_type] = chosen
            last = chosen["line_index"]
        else:
            warnings.append(f"missing_marker:{marker_type}")
    return selected, warnings


def slice_sections(lines, markers):
    n = len(lines)
    warnings = []
    idx = {k: int(v["line_index"]) for k, v in markers.items() if k in MARKER_ORDER}
    if "preamble" not in idx:
        idx["preamble"] = 0
        warnings.append("fallback_start:preamble")
    if "noi_dung_vu_an" not in idx:
        idx["noi_dung_vu_an"] = max(idx["preamble"] + 1, int(n * 0.15))
        warnings.append("fallback_boundary:noi_dung_vu_an")
    if "dai_dien_vks" not in idx:
        idx["dai_dien_vks"] = idx.get("nhan_dinh", max(idx["noi_dung_vu_an"], int(n * 0.48)))
        warnings.append("empty_or_missing:dai_dien_vks")
    if "nhan_dinh" not in idx:
        idx["nhan_dinh"] = max(idx["dai_dien_vks"], int(n * 0.58))
        warnings.append("fallback_boundary:nhan_dinh")
    if "quyet_dinh" not in idx:
        idx["quyet_dinh"] = max(idx["nhan_dinh"] + 1, int(n * 0.78))
        warnings.append("fallback_boundary:quyet_dinh")
    if not (idx["preamble"] <= idx["noi_dung_vu_an"] <= idx["dai_dien_vks"] <= idx["nhan_dinh"] <= idx["quyet_dinh"] <= n):
        warnings.append("invalid_order_ratio_fallback")
        idx["preamble"] = 0
        idx["noi_dung_vu_an"] = max(1, int(n * 0.15))
        idx["dai_dien_vks"] = max(idx["noi_dung_vu_an"], int(n * 0.48))
        idx["nhan_dinh"] = max(idx["dai_dien_vks"], int(n * 0.58))
        idx["quyet_dinh"] = max(idx["nhan_dinh"] + 1, int(n * 0.78))
    sections = {
        "preamble": "\n".join(lines[idx["preamble"]:idx["noi_dung_vu_an"]]).strip(),
        "noi_dung_vu_an": "\n".join(lines[idx["noi_dung_vu_an"]:idx["dai_dien_vks"]]).strip(),
        "dai_dien_vks": "\n".join(lines[idx["dai_dien_vks"]:idx["nhan_dinh"]]).strip(),
        "nhan_dinh": "\n".join(lines[idx["nhan_dinh"]:idx["quyet_dinh"]]).strip(),
        "quyet_dinh": "\n".join(lines[idx["quyet_dinh"]:]).strip()
    }
    for k, v in sections.items():
        if not v:
            warnings.append(f"empty_section:{k}")
    return sections, warnings


def predict_file(path: Path, model, config, word2idx, char2idx, tag2idx, idx2tag, device, threshold, include_predictions):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = compact_nonempty_lines(split_lines(raw))
    if not lines:
        return {"file_name": path.name, "sections_json": {k: "" for k in MARKER_ORDER}, "selected_markers": {}, "warnings": ["empty_file"]}
    word_ids, char_ids, feats, lengths, mask = encode_lines(lines, word2idx, char2idx, config, device)
    with torch.no_grad():
        emissions = model.emissions(word_ids, char_ids, feats, lengths)
        probs = torch.softmax(emissions, dim=-1)[0, :len(lines)].cpu().tolist()
        path_ids = model.crf.decode(emissions, mask)[0]
    decoded_tags = [idx2tag[int(x)] for x in path_ids[:len(lines)]]
    selected, warnings_1 = choose_markers(lines, decoded_tags, probs, tag2idx, threshold)
    sections, warnings_2 = slice_sections(lines, selected)
    out = {
        "file_name": path.name,
        "selected_markers": selected,
        "warnings": warnings_1 + warnings_2,
        "sections_json": sections
    }
    if include_predictions:
        out["line_predictions"] = [{"line_index": i, "line": lines[i], "tag": decoded_tags[i], "scores": {m: probs[i][tag2idx[m]] for m in MARKER_ORDER}} for i in range(len(lines))]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--input_folder", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--include_predictions", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=0)
    args = parser.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, config, word2idx, char2idx, tag2idx, idx2tag = load_bundle(Path(args.model_dir), device)
    if args.input_file:
        rows = [predict_file(args.input_file, model, config, word2idx, char2idx, tag2idx, idx2tag, device, args.threshold, args.include_predictions)]
    elif args.input_folder:
        rows = [predict_file(p, model, config, word2idx, char2idx, tag2idx, idx2tag, device, args.threshold, args.include_predictions) for p in sorted(args.input_folder.glob("*.md"))]
    else:
        raise ValueError("input_file or input_folder is required")
    text = json.dumps(rows[0] if args.input_file else rows, ensure_ascii=False, indent=2)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
