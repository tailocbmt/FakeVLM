python scripts/predict.py \
  --model_path "lingcco/fakeVLM" \
  --val_batch_size 16 \
  --workers 16 \
  --output_path results/fakevlm.json \
  --test_json_file "./evons_data/test_multilabel.csv" \
  --data_base_test "./evons_data" \