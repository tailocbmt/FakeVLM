python scripts/predict.py \
  --model_path "lingcco/fakeVLM" \
  --val_batch_size 16 \
  --workers 16 \
  --output_path results/fakevlm_vision.json \
  --test_json_file "../aigen-foodreview/evons_data/test_multilabel.csv" \
  --data_base_test "../aigen-foodreview/evons_data" \