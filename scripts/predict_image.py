import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from transformers import AutoModelForImageClassification, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Legion Model Training")

    # Model-specific settings
    parser.add_argument(
        "--model_path", default="Smogy/SMOGY-Ai-images-detector", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--data_base_test", default="", type=str)
    parser.add_argument("--test_json_file", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    return parser.parse_args()


class legion_cls_dataset(Dataset):
    def __init__(self, args, cfg):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.data = pd.read_csv(args.test_json_file)

        self.processor = AutoTokenizer.from_pretrained(
            args.model_path)   # tokenizer ships in this repo

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_path = os.path.join(
            self.args.data_base_test, item['image_path'])

        image = Image.open(img_path).convert('RGB').resize((256, 256))
        label = item['label_image']

        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True
        )

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        return inputs, label


def calculate_results_acc(results):
    acc_results = {}

    for cate in results:
        data = results[cate]

        right_real = data['right']['right_real']
        right_fake = data['right']['right_fake']
        wrong_real = data['wrong']['wrong_real']
        wrong_fake = data['wrong']['wrong_fake']

        total_real = right_real + wrong_real
        total_fake = right_fake + wrong_fake
        total = total_real + total_fake

        acc_total = (right_real + right_fake) / total if total != 0 else 0
        acc_real = right_real / total_real if total_real != 0 else 0
        acc_fake = right_fake / total_fake if total_fake != 0 else 0

        acc_results[cate] = {
            'total_samples': total,
            'total_accuracy': round(acc_total, 4),
            'real_accuracy': round(acc_real, 4),
            'fake_accuracy': round(acc_fake, 4),
            'confusion_matrix': {
                'right_real': right_real,
                'wrong_real': wrong_real,
                'right_fake': right_fake,
                'wrong_fake': wrong_fake,
            }
        }

    global_stats = {
        'total_right': sum(r['right']['right_real'] + r['right']['right_fake'] for r in results.values()),
        'total_wrong': sum(r['wrong']['wrong_real'] + r['wrong']['wrong_fake'] for r in results.values())
    }
    global_stats['global_accuracy'] = global_stats['total_right'] / \
        (global_stats['total_right'] + global_stats['total_wrong'])

    return {
        'category_acc': acc_results,
        'global_stats': global_stats
    }


def validate(args, model, cls_test_dataloader, device):
    # Initialize the results dictionary matching what calculate_results_acc expects.
    # Assuming a single 'all' category for the entire dataset.
    outputs = []
    results = {
        'all': {
            'right': {'right_real': 0, 'right_fake': 0},
            'wrong': {'wrong_real': 0, 'wrong_fake': 0}
        }
    }

    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(cls_test_dataloader, desc="Evaluating"):
            # Move inputs to device and squeeze the extra dimension that HuggingFace
            # tokenizers sometimes add when return_tensors="pt" is combined with DataLoader
            input_ids = inputs["input_ids"].squeeze(1).to(device)

            # Get logits and calculate probabilities
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1)

            # Get the predicted class (0 for Real, 1 for Fake)
            # using argmax across the batch
            preds = torch.argmax(probs, dim=-1)

            # Iterate over the batch to populate the results dictionary
            for i in range(len(labels)):
                pred = preds[i].item()

                # IMPORTANT: Adjust this based on how your labels are stored in the CSV.
                # Here we assume labels are integers: 0 for real, 1 for fake.
                # If they are strings, you might need: int(labels[i] == 'fake')
                true_label = int(labels[i])
                # The raw "MELD score" probability
                prob_fake = probs[i, 1].item()

                outputs.append({
                    'label': true_label,
                    'pred': pred,
                    'prob_fake': prob_fake
                })

                if true_label == 0:  # Ground Truth is Real
                    if pred == 0:
                        results['all']['right']['right_real'] += 1
                    else:
                        results['all']['wrong']['wrong_real'] += 1
                elif true_label == 1:  # Ground Truth is Fake
                    if pred == 1:
                        results['all']['right']['right_fake'] += 1
                    else:
                        results['all']['wrong']['wrong_fake'] += 1

    # Calculate and return final metrics using your helper function
    metrics = calculate_results_acc(results)

    os.makedirs('text_results', exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(outputs, file, indent=2)

    acc = calculate_results_acc(results)
    print(metrics)
    print(acc)

    return metrics, outputs


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(args.model)
    model.eval().to(device)

    cls_test_dataset = legion_cls_dataset(args, cfg=cfg)
    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    validate(args, model, cls_test_dataloader, device)


if __name__ == "__main__":
    main()
