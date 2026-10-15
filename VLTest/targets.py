
from __future__ import annotations

import json
from pathlib import Path


def make_preprocessed_key(task, entry, *, right_image_path=None):
    import hashlib

    import torch
    import torchvision.transforms as T
    from PIL import Image

    task, name = task.lower(), entry["name"].lower()
    vr_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    vqa_transform = T.Compose([
        T.Resize((480, 480), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def serialize(values):
        digest = hashlib.blake2b(digest_size=16)
        for label, tensor in sorted(values.items()):
            tensor = tensor.detach().contiguous().cpu()
            digest.update(label.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.digest()

    def key(image_path, text):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if name == "beit3":
            max_len = 40 if task in {"vr", "nlvr"} else 64
            tokens = entry["tokenizer"](
                text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_len)
            transform = vr_transform if task in {"vr", "nlvr"} else vqa_transform
            values = {"image": transform(image),
                      "tokens": tokens["input_ids"],
                      "mask": tokens["attention_mask"]}
            if task in {"vr", "nlvr"}:
                if right_image_path is None:
                    raise ValueError("NLVR2 requires the fixed right image")
                with Image.open(right_image_path) as opened:
                    values["right"] = vr_transform(opened.convert("RGB"))
            return serialize(values)
        if task == "vqa" and name == "paligemma":
            prompt = f"<image> question: {text} short answer:"
            inputs = entry["processor"](text=prompt, images=image,
                                          return_tensors="pt")
        elif task in {"vr", "nlvr"}:
            if right_image_path is None:
                raise ValueError("NLVR2 requires the fixed right image")
            with Image.open(right_image_path) as opened:
                right = opened.convert("RGB")
            inputs = entry["processor"](
                [image, right], text, return_tensors="pt",
                truncation=True, max_length=40)
        else:
            inputs = entry["processor"](image, text, return_tensors="pt")
        return serialize({name: value for name, value in inputs.items()
                          if isinstance(value, torch.Tensor)})

    return key


def load_paper_model(task, name, device, *, assets_dir="."):
    import torch
    from transformers import (
        AutoProcessor,
        BlipForImageTextRetrieval,
        BlipProcessor,
        PaliGemmaForConditionalGeneration,
        ViltForImageAndTextRetrieval,
        ViltForImagesAndTextClassification,
        ViltProcessor,
        XLMRobertaTokenizer,
    )

    task, name = task.lower(), name.lower()
    assets = Path(assets_dir)
    if task == "itm" and name == "vilt":
        checkpoint = "dandelin/vilt-b32-finetuned-coco"
        return {"name": name, "processor": ViltProcessor.from_pretrained(checkpoint),
                "model": ViltForImageAndTextRetrieval.from_pretrained(
                    checkpoint).to(device).eval()}
    if task == "itm" and name == "blip":
        checkpoint = "Salesforce/blip-itm-base-coco"
        return {"name": name, "processor": BlipProcessor.from_pretrained(checkpoint),
                "model": BlipForImageTextRetrieval.from_pretrained(
                    checkpoint).to(device).eval()}
    if task in {"vr", "nlvr"} and name == "vilt":
        checkpoint = "dandelin/vilt-b32-finetuned-nlvr2"
        return {"name": name, "processor": ViltProcessor.from_pretrained(checkpoint),
                "model": ViltForImagesAndTextClassification.from_pretrained(
                    checkpoint).to(device).eval()}
    if task == "vqa" and name == "paligemma":
        checkpoint = "google/paligemma-3b-ft-vqav2-224"
        return {"name": name, "processor": AutoProcessor.from_pretrained(checkpoint),
                "model": PaliGemmaForConditionalGeneration.from_pretrained(
                    checkpoint, torch_dtype=torch.float16 if str(device).startswith("cuda")
                    else torch.float32).to(device).eval()}
    if name == "beit3" and task in {"vr", "nlvr", "vqa"}:
        from beit3.modeling_finetune import (
            beit3_large_patch16_224_nlvr2,
            beit3_large_patch16_480_vqav2,
        )
        tokenizer = XLMRobertaTokenizer(str(assets / "beit3.spm"))
        if task in {"vr", "nlvr"}:
            model = beit3_large_patch16_224_nlvr2(pretrained=False)
            checkpoint = assets / "beit3_large_patch16_224_nlvr2.pth"
            state = torch.load(checkpoint, map_location="cpu")["model"]
            model.load_state_dict(state, strict=True)
            return {"name": name, "tokenizer": tokenizer,
                    "model": model.to(device).eval()}
        model = beit3_large_patch16_480_vqav2(pretrained=False)
        checkpoint = assets / "beit3_large_patch16_480_vqa.pth"
        state = torch.load(checkpoint, map_location="cpu")["model"]
        model.load_state_dict(state, strict=True)
        vocabulary = assets / "data" / "vqav2" / "answer2label.txt"
        with vocabulary.open(encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
        if any(int(row["label"]) != index for index, row in enumerate(entries)):
            raise ValueError("BEiT-3 answer vocabulary has nonsequential labels")
        label2ans = [row["answer"] for row in entries]
        return {"name": name, "tokenizer": tokenizer,
                "label2ans": label2ans, "model": model.to(device).eval()}
    raise ValueError(f"unsupported task/model combination: {task}/{name}")


def make_target(task, entry, device, *, right_image_path=None):
    import torch
    import torchvision.transforms as T
    from PIL import Image

    name = entry["name"].lower()
    task = task.lower()
    model = entry["model"]
    model.eval()

    beit3_vr_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    beit3_vqa_transform = T.Compose([
        T.Resize((480, 480), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    @torch.no_grad()
    def target(image_path, text):
        image = Image.open(image_path).convert("RGB")
        if task == "itm":
            inputs = entry["processor"](image, text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            if name == "vilt":
                return int(outputs.logits.reshape(-1)[0].item() > 0)
            if name == "blip":
                return int(outputs.itm_score.argmax(dim=-1).item())

        if task in {"vr", "nlvr"}:
            if right_image_path is None:
                raise ValueError("NLVR2 requires the fixed right image")
            right = Image.open(right_image_path).convert("RGB")
            if name == "vilt":
                inputs = entry["processor"](
                    [image, right], text, return_tensors="pt",
                    truncation=True, max_length=40)
                outputs = model(
                    input_ids=inputs.input_ids.to(device),
                    pixel_values=inputs.pixel_values.unsqueeze(0).to(device))
                return int(outputs.logits.argmax(dim=-1).item())
            if name == "beit3":
                tokens = entry["tokenizer"](
                    text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=40)
                logits = model(
                    beit3_vr_transform(image).unsqueeze(0).to(device),
                    beit3_vr_transform(right).unsqueeze(0).to(device),
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].eq(0).to(device))
                return int(logits.argmax(dim=-1).item())

        if task == "vqa":
            if name == "paligemma":
                prompt = f"<image> question: {text} short answer:"
                inputs = entry["processor"](
                    text=prompt, images=image, return_tensors="pt").to(device)
                length = inputs["input_ids"].shape[-1]
                generated = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False, num_beams=1)
                return entry["processor"].decode(
                    generated[0][length:], skip_special_tokens=True).strip()
            if name == "beit3":
                tokens = entry["tokenizer"](
                    text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=64)
                logits = model(
                    beit3_vqa_transform(image).unsqueeze(0).to(device),
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].eq(0).to(device))
                index = int(logits.argmax(dim=-1).item())
                return entry["label2ans"][index]
            if name == "blip2":
                prompt = f"Question: {text}\nAnswer with ONE word or ONE number only."
                inputs = entry["processor"](
                    images=image, text=prompt, return_tensors="pt").to(device)
                output = model.generate(
                    **inputs, max_new_tokens=4, do_sample=False, num_beams=1)
                return entry["processor"].batch_decode(
                    output, skip_special_tokens=True)[0].strip()
        raise ValueError(f"unsupported task/model combination: {task}/{name}")

    return target
