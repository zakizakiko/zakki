import bridge_bid_transformer
# 学習（小さめのLMでOK）
python bridge_bid_transformer.py --mode train \
  --model distilgpt2 \
  --train_path train.jsonl \
  --out_dir bridge-bid-sa

# 推論
python bridge_bid_transformer.py --mode predict \
  --example_json sample_case.json \
  --out_dir bridge-bid-sa
