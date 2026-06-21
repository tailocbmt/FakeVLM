python scripts/predict_text.py \
  --model_path "anon-review-meld-2026/meld" \
  --val_batch_size 256 \
  --workers 16 \
  --output_path results/fakevlm_text.json \
  --test_json_file "../aigen-foodreview/evons_data/test_multilabel.csv" \