import json
import random
import os

QUESTION_PATH = "annotations/v2_val2014_questions.json"
ANNOTATION_PATH = "annotations/v2_val2014_annotations.json"
OUTPUT_PATH = "image2captions.json"

# --------------- train ------------------
# QUESTION_PATH = "annotations/v2_train2014_questions.json"
# ANNOTATION_PATH = "annotations/v2_train2014_annotations.json"
# OUTPUT_PATH = "train.json"
EXCLUDE_JSON_PATH = "image2captions.json"
# --------------- train ------------------

TARGET_NUM_IMAGES = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def coco_image_name(image_id: int) -> str:
    return f"{image_id:012d}.jpg"


def main():
    # 1. load json
    with open(QUESTION_PATH, "r", encoding="utf-8") as f:
        questions_json = json.load(f)

    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations_json = json.load(f)

    questions = questions_json["questions"]
    annotations = annotations_json["annotations"]

    # 2. build question_id -> question text
    qid_to_question = {
        q["question_id"]: (q["image_id"], q["question"]) for q in questions
    }

    # 3. build image_id -> list of (question, answers)
    image_to_pairs = {}

    for ann in annotations:
        qid = ann["question_id"]
        if qid not in qid_to_question:
            continue

        image_id, question_text = qid_to_question[qid]
        answers = ann["answers"]

        if image_id not in image_to_pairs:
            image_to_pairs[image_id] = []

        image_to_pairs[image_id].append({"question": question_text, "answers": answers})

    # 4. sample 100 unique image_ids
    # --------------- train ------------------
    all_image_ids = [
        image_id
        for image_id in image_to_pairs.keys()
        if coco_image_name(image_id) not in exclude_images
    ]
    # --------------- train ------------------
    # all_image_ids = list(image_to_pairs.keys())
    if len(all_image_ids) < TARGET_NUM_IMAGES:
        raise ValueError(f"Only {len(all_image_ids)} unique images found!")

    sampled_image_ids = random.sample(all_image_ids, TARGET_NUM_IMAGES)

    # 5. for each image, take ONE question-answer pair
    output = {}
    for image_id in sampled_image_ids:
        pair = image_to_pairs[image_id][0]  # take first pair
        img_name = coco_image_name(image_id)
        output[img_name] = pair

    # 6. save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output)} image-question pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    # --------------- train ------------------
    exclude_images = set()
    if EXCLUDE_JSON_PATH and os.path.exists(EXCLUDE_JSON_PATH):
        with open(EXCLUDE_JSON_PATH, "r", encoding="utf-8") as f:
            exclude_images = set(json.load(f).keys())
    # --------------- train ------------------
    main()
