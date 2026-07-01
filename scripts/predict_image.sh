python scripts/predict_image.py \
  --model_path "Bombek1/ai-image-detector-siglip-dinov2" \
  --val_batch_size 256 \
  --workers 16 \
  --output_path results/fakevlm_image.json \
  --test_json_file "../aigen-foodreview/evons_data/test_multilabel.csv" \
  --data_base_test "../aigen-foodreview/evons_data" \