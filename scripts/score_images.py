"""
Script to score images using aesthetic-predictor-v2-5.
Reads a JSON array of image paths from stdin.
Outputs progress lines to stdout in the format: PROGRESS:scored:total
Scores are saved as CSV files alongside the images.
"""
import sys
import json
import os
import csv

def get_csv_path(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    return base + '.csv'

def has_score(img_path: str, metric_key: str) -> bool:
    csv_path = get_csv_path(img_path)
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1 and row[0] == metric_key:
                    return True
    except Exception:
        pass
    return False

def save_score(img_path: str, metric_key: str, value: float) -> None:
    csv_path = get_csv_path(img_path)
    # Read existing scores if any
    existing = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0] != metric_key:
                        existing.append(row)
        except Exception:
            pass
    existing.append([metric_key, str(value)])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(existing)

def main():
    data = json.load(sys.stdin)
    image_paths = data.get('images', [])
    metric_key = 'aesthetic-predictor-v2-5'

    # Filter to images that don't already have a score
    to_score = [p for p in image_paths if not has_score(p, metric_key)]
    total = len(to_score)
    scored = 0

    if total == 0:
        print(f'PROGRESS:{len(image_paths)}:{len(image_paths)}', flush=True)
        return

    # Load model only if there are images to score
    try:
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        from PIL import Image
    except ImportError as e:
        print(f'ERROR:Missing dependency: {e}', flush=True)
        sys.exit(1)

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(torch.bfloat16).to(device)
    model.eval()

    for img_path in to_score:
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = preprocessor(images=image, return_tensors='pt').to(torch.bfloat16).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            score = outputs.logits.squeeze().float().item()
            save_score(img_path, metric_key, score)
        except Exception as e:
            print(f'WARN:Failed to score {img_path}: {e}', flush=True)
        scored += 1
        print(f'PROGRESS:{scored}:{total}', flush=True)

if __name__ == '__main__':
    main()
