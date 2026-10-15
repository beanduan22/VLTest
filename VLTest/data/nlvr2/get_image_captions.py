import json
import random
import os
import requests

from PIL import Image

RANDOM_SEED = 123456
random.seed(RANDOM_SEED)

# ========== config ==========
# DEV_JSON_PATH = "annotations/train.json"
# LEFT_DIR = "train/images_left"
# RIGHT_DIR = "train/images_right"
# RECORD_JSON = "train.json"
DEV_JSON_PATH = "annotations/dev.json"
LEFT_DIR = "val/images_left"
RIGHT_DIR = "val/images_right"
RECORD_JSON = "image2captions.json"

TARGET_NUM = 100
TIMEOUT = 10

# ========== Initialize ==========
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

if os.path.exists(RECORD_JSON):
    with open(RECORD_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
else:
    records = {}

current_id = len(records) + 1


# ========== Utility functions ==========


def is_image_decodable(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.close()
        return True
    except Exception:
        return False


def format_id(i):
    return str(i).zfill(12)


def can_download(url):
    try:
        r = requests.get(url, timeout=TIMEOUT, stream=True)
        return r.status_code == 200
    except Exception:
        return False


def download_image(url, save_path):
    r = requests.get(url, timeout=TIMEOUT, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


# ========== Main logic ==========
def collect_random_samples():
    global current_id

    with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    random.shuffle(samples)  # Random sort

    print(f"[INFO] Total dev samples: {len(samples)}")

    for sample in samples:
        if len(records) >= TARGET_NUM:
            break

        left_url = sample["left_url"]
        right_url = sample["right_url"]
        sentence = sample["sentence"]
        ans = list(sample["validation"].values())[0]

        # Verify downloadability
        if not can_download(left_url) or not can_download(right_url):
            continue

        img_id = format_id(current_id)
        left_path = os.path.join(LEFT_DIR, f"{img_id}.jpg")
        right_path = os.path.join(RIGHT_DIR, f"{img_id}.jpg")

        try:
            download_image(left_url, left_path)
            download_image(right_url, right_path)
        except Exception:
            # Clean up failed remnants
            if os.path.exists(left_path):
                os.remove(left_path)
            if os.path.exists(right_path):
                os.remove(right_path)
            continue

        # ===== PIL decode validation）=====
        if not is_image_decodable(left_path) or not is_image_decodable(right_path):
            if os.path.exists(left_path):
                os.remove(left_path)
            if os.path.exists(right_path):
                os.remove(right_path)
            continue

        # Record sentence
        records[f"{img_id}.jpg"] = {"question": sentence, "answers": ans}
        current_id += 1

        print(f"[OK] {len(records)}/{TARGET_NUM} saved → ID {img_id}")

    with open(RECORD_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Collected {len(records)} samples.")


if __name__ == "__main__":
    collect_random_samples()
