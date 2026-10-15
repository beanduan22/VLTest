import json
import os
from collections import defaultdict
import random

IMAGE_DIR = "val"
CAPTION_JSON = "annotations/captions_val2017.json"
OUTPUT_JSON = "image2captions.json"

# ========= 1. Read images in current directory =========
image_files = set([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"[INFO] Found {len(image_files)} images in {IMAGE_DIR}")

# ========= 2. Load caption json =========
with open(CAPTION_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

# ========= 3. Set image_id -> file_name  =========
id2file = {}
for img in coco["images"]:
    if img["file_name"] in image_files:
        id2file[img["id"]] = img["file_name"]

print(f"[INFO] Matched {len(id2file)} image ids")

# ========= 4. collect captions =========
image2captions = defaultdict(list)

for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id in id2file:
        fname = id2file[img_id]
        if len(image2captions[fname]) < 5:
            image2captions[fname].append(ann["caption"])

# ========= 5. sanity check =========
for k, v in image2captions.items():
    if len(v) != 5:
        print(f"[WARN] {k} has {len(v)} captions")

# ========= 5.5 random select 1 caption per image =========
random.seed(123456)

image2onecaption = {}

for fname, caps in image2captions.items():
    if len(caps) == 0:
        continue
    final_caption = random.choice(caps)
    image2onecaption[fname] = {"question": final_caption, "answers": True}

# ========= 6. save =========
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(image2onecaption, f, indent=4, ensure_ascii=False)

print(f"[DONE] Saved to {OUTPUT_JSON}")
