import os
import sys
import time
import platform
import argparse
import random

import json
from collections import Counter

import torch
import numpy as np
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from omegaconf import OmegaConf
import PIL
from PIL import Image
from taming.models.vqgan import VQModel
from main import instantiate_from_config
from tqdm import tqdm
from glob import glob
from queue import Queue
from threading import Thread

# Ensure local imports work
sys.path.append(".")

# Disable gradients to save memory
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda")

# Hyperparameters
DATA_ROOT = "data"
TASK_REGISTRY = {
    "itm": {
        "name": "image_text_matching",
        "dataset": "coco",
        "models": ["albef", "blip"],
    },
    "vqa": {
        "name": "visual_question_answering",
        "dataset": "vqav2",
        "models": ["blip2", "lxmert"],
    },
    "nlvr": {
        "name": "visual_reasoning",
        "dataset": "nlvr2",
        "models": ["vilbert", "lxmert"],
    },
}


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imgid_to_key(x):
    res = ""
    if isinstance(x, str):
        x = os.path.basename(x)  # Extract filename
        x = os.path.splitext(x)[0]  # Remove .jpg / .png
    res = f"{int(x):012d}.jpg"
    return res


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


def norm01(img):
    img = (img + 1.0) / 2.0
    return img


def preprocess(img, target_image_size=256):
    s = min(img.size)  # shortest side

    # Just allow upscaling instead of raising
    # (for larger images this will downscale; for smaller, it will upscale)
    r = target_image_size / s
    new_h = round(r * img.size[1])
    new_w = round(r * img.size[0])

    img = TF.resize(img, (new_h, new_w), interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img


save_queue = Queue(maxsize=512)


def saver():
    while True:
        item = save_queue.get()
        if item is None:
            save_queue.task_done()
            break
        img, path = item
        save_image(img, path)
        save_queue.task_done()


save_thread = Thread(target=saver)
save_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUN THE BASELINE CONFIGURATION")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASK_REGISTRY.keys(),
        help=(
            "Testing task:\n"
            "  itm  : Image-Text Matching (mscoco, CLIP vs BLIP)\n"
            "  vqa  : Visual Question Answering (vqav2, BLIP2 vs LXMERT)\n"
            "  nlvr : Visual Reasoning (nlvr2, ViLBERT vs LXMERT)"
        ),
    )
    parser.add_argument(
        "--seed_nums", type=int, default=50, help="Number of seeds pool"
    )

    parser.add_argument(
        "--pre_seed_num",
        type=int,
        default=10,
        help="Number of initial seeds used to generate perturbations",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
        help="Directory to save all probe outputs and logs",
    )

    parser.add_argument(
        "--pre_level_start",
        type=float,
        default=0.01,
        help="Starting perturbation strength level",
    )

    parser.add_argument(
        "--pre_level_step",
        type=float,
        default=0.01,
        help="Step size for increasing perturbation strength",
    )

    parser.add_argument(
        "--pre_time",
        type=int,
        default=25,
        help="Number of perturbation iterations per sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=True,
        help="random seed for initialization",
    )

    # Parse arguments from the command line
    args = parser.parse_args()
    set_seed(args.seed)

    # ===== Global configuration =====
    SEED_NUMS = args.seed_nums
    PRE_SEED_NUM = args.pre_seed_num
    OUTPUT_PATH = args.output_path
    REF_SEED_NUM = SEED_NUMS - PRE_SEED_NUM
    PRE_LEVEL_START = args.pre_level_start
    PRE_LEVEL_STEP = args.pre_level_step
    PRE_TIME = args.pre_time
    PER_METHOD = "image"
    CAPTION = "NO CAPTION, BASELINE"

    # Build the task context
    task = TASK_REGISTRY[args.task]
    ctx = {
        "task": args.task,
        "task_name": task["name"],
        "dataset": task["dataset"],
        "dataset_path": os.path.join(DATA_ROOT, task["dataset"]),
        "models": task["models"],
    }

    print("\n====== Task Context ======")
    print(f" Task    : {ctx['task_name']}")
    print(f" Dataset : {ctx['dataset']}")
    print(f" Models  : {ctx['models'][0]}  vs  {ctx['models'][1]}")
    print("==========================\n")

    print("[VQGAN] Loaded checkpoint...")
    # VQGAN models load

    vqgan_cfg_path = "configs/coco_scene_images_transformer.yaml"
    vqgan_ckpt_path = "logs/coco_epoch117.ckpt"
    cfg = OmegaConf.load(vqgan_cfg_path)
    first_stage_cfg = cfg.model.params.first_stage_config
    first_stage_cfg.params.ckpt_path = vqgan_ckpt_path
    vqgan = instantiate_from_config(first_stage_cfg).eval().to(DEVICE)

    # VQGAN - Load checkpoint weights
    sd = torch.load(vqgan_ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = vqgan.load_state_dict(sd, strict=False)
    print("[VQGAN] Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    vqgan = vqgan.to(DEVICE).eval()

    # IMAGE Load
    IMAGE_DATA_PATH = os.path.join(DATA_ROOT, ctx["dataset"], "val")

    image_paths = sorted(
        glob(os.path.join(IMAGE_DATA_PATH, "**", "*.jpg"), recursive=True)
    )
    image_paths += sorted(
        glob(os.path.join(IMAGE_DATA_PATH, "**", "*.png"), recursive=True)
    )

    prep = T.Compose(
        [
            T.Resize(256, interpolation=Image.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
        ]
    )
    print(len(image_paths), "images found.")

    # TEXT Load
    caption_path = os.path.join(DATA_ROOT, task["dataset"], "image2captions.json")

    with open(caption_path, "r", encoding="utf-8") as f:
        caption_dict = json.load(f)

    # output folder Load
    dir_name = f"{OUTPUT_PATH}/{ctx['task']}-{PER_METHOD}-baseline-{REF_SEED_NUM}-{PRE_TIME}-{PRE_LEVEL_START}-{PRE_LEVEL_STEP}-seed{PRE_SEED_NUM}"
    os.makedirs(dir_name, exist_ok=True)
    event_image_path = os.path.join(dir_name, f"images.jsonl")
    event_image = open(event_image_path, "w", encoding="utf-8")
    event_text_path = os.path.join(dir_name, f"texts.jsonl")
    event_text = open(event_text_path, "w", encoding="utf-8")

    start_time = time.time()

    # Get the vector library
    vector_lib = []
    for img_path in list(image_paths)[PRE_SEED_NUM:SEED_NUMS]:
        img_name = os.path.basename(img_path).split(".")[0]
        img = Image.open(img_path).convert("RGB")
        x_vqgan = preprocess(img)
        x = preprocess_vqgan(x_vqgan).to(DEVICE)
        z_q, _, [_, _, indices] = vqgan.encode(x)
        vector_lib.append((img_name, z_q))

    for count, img_path in enumerate(
        tqdm(list(image_paths)[:PRE_SEED_NUM], desc="Processing images")
    ):
        img_name = os.path.basename(img_path).split(".")[0]
        subdir_name = f"{dir_name}/{img_name}"
        os.makedirs(subdir_name, exist_ok=True)

        img_key = imgid_to_key(img_path)
        if img_key not in caption_dict:
            raise KeyError(f"Caption not found for image {img_key}")
        orig_caption = caption_dict[img_key]["question"]

        # Preprocess image
        img = Image.open(img_path).convert("RGB")
        x_vqgan = preprocess(img)
        x = preprocess_vqgan(x_vqgan).to(DEVICE)
        z_q, _, [_, _, indices] = vqgan.encode(x)
        # latent_grid = indices.reshape(16, 16)
        # print(latent_grid)
        old_set = set(indices.view(-1).cpu().tolist())

        # Save undoctored reconstruction
        orig_image = norm01(vqgan.decode(z_q))
        path = f"{subdir_name}/0_orig_image.png"
        save_image(orig_image, path)

        # fallback: only seed text
        orig_image_event = {
            "seed_id": count,
            "seed_name": img_name,
            "image_path": path,
            "ori_tag": True,
        }
        event_image.write(json.dumps(orig_image_event) + "\n")

        # fallback: only seed text
        text_event = {
            "seed_id": count,
            "seed_name": img_name,
            "seed_text": orig_caption,
        }

        event_text.write(json.dumps(text_event) + "\n")

        for index, ref in enumerate(vector_lib):
            for i in range(PRE_TIME):
                level = PRE_LEVEL_START + i * PRE_LEVEL_STEP
                ori_latent = z_q.clone()
                cur_latent = ori_latent + (level * ref[1])

                rc = norm01(vqgan.decode(cur_latent))

                cur_z_q, _, [_, _, cur_indices] = vqgan.encode(rc)
                new_set = set(cur_indices.view(-1).cpu().tolist())

                new_only = new_set - old_set

                pic_name = f"No{index+1}_{ref[0]}_{i+1}_pert_{level:.4f}"
                path = f"{subdir_name}/{pic_name}.png"
                save_queue.put((rc, path))

                image_event = {
                    "seed_id": count,
                    "seed_name": img_name,
                    "image_path": path,
                    "ref_image": ref[0],
                    "level": level,
                    "dst_codeword": sorted(list(new_only)),
                }
                event_image.write(json.dumps(image_event) + "\n")

    # ===============================
    # Flush image saving queue
    # ===============================
    print("[Saver] Waiting for all images to be saved...")
    save_queue.join()  # Wait for all save_image to complete

    print("[Saver] Shutting down saver thread...")
    save_queue.put(None)  # Notify saver to exit
    save_thread.join()  # Wait for saver to fully terminate
    print("[Saver] Saver thread exited cleanly.")
    # ===============================
    end_time = time.time()
    event_image.close()
    event_text.close()

    time_taken = end_time - start_time
    time_per_seed = time_taken / PRE_SEED_NUM

    print(f"ALL seeds time cost: {time_taken:.4f}s")
    print(f"PER seed time cost: {time_per_seed:.4f}s")
    print("========================================")

    # ===============================
    # Save timing summary (minimal)
    # ===============================
    timing_summary = {
        "total_time_sec": round(time_taken, 4),
        "avg_time_per_seed_sec": round(time_per_seed, 4),
    }

    timing_path = os.path.join(dir_name, "timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"[Timing] Saved timing summary to {timing_path}")
