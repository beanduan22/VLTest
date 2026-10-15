import json
import os
import subprocess

# JSON_PATH = "image2captions.json"
# SAVE_DIR = "val"

JSON_PATH = "train.json"
SAVE_DIR = "train"

COCO_TRAIN_URL = "http://images.cocodataset.org/train2017"
COCO_VAL_URL = "http://images.cocodataset.org/val2017"

os.makedirs(SAVE_DIR, exist_ok=True)


def wget(url, save_path):
    cmd = ["wget", "-q", "-O", save_path, url]
    return subprocess.run(cmd, check=False)


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = downloaded = skipped = failed = 0

    for img_name in data.keys():
        total += 1
        save_path = os.path.join(SAVE_DIR, img_name)

        if os.path.exists(save_path):
            skipped += 1
            print(f"[SKIP] {img_name} already exists")
            continue

        # 1️⃣ try train2017
        train_url = f"{COCO_TRAIN_URL}/{img_name}"
        print(f"[TRY train2017] {img_name}")
        ret = wget(train_url, save_path)

        success = ret.returncode == 0 and os.path.exists(save_path)

        # 2️⃣ fallback to val2017
        if not success:
            if os.path.exists(save_path):
                os.remove(save_path)

            val_url = f"{COCO_VAL_URL}/{img_name}"
            print(f"[TRY val2017] {img_name}")
            ret = wget(val_url, save_path)
            success = ret.returncode == 0 and os.path.exists(save_path)

        if success:
            downloaded += 1
        else:
            failed += 1
            print(f"[FAILED] {img_name}")

    print("\n===== Download Summary =====")
    print(f"Total images     : {total}")
    print(f"Downloaded       : {downloaded}")
    print(f"Skipped (exists) : {skipped}")
    print(f"Failed           : {failed}")
    print("============================")
    print("Download finished")


if __name__ == "__main__":
    main()
