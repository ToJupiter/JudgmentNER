#  NER on Vietnamese court judgment with CNN:
1. As a replacement for complex large language model with high inference time, we utilize the labelling strategy with DeepSeek LLM and then train a small CNN model with inference time of a few miliseconds for the regex-ed input.

# Training and inference with the BiLSTM + CharCNN model:
The command for training the model is presented here:
``` bash
python cnn_training.py \
  --jsonl_files data/train.jsonl \
  --law_csv data/laws.csv \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.001 \
  --output_dir artifacts/legal_ner \
  --min_freq 1
```

Inference:
``` bash
python cnn_training.py \
  --infer \
  --model_dir artifacts/legal_ner \
  --text "Căn cứ điểm a khoản 1 Điều 355, Điều 356 Bộ luật Tố tụng hình sự"
  --cpu
```