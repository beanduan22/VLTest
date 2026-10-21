
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def make_preprocessed_key(task, entry, *, right_image_path=None):
    import hashlib

    import torch
    import torchvision.transforms as T
    from PIL import Image

    task, name = task.lower(), entry["name"].lower()
    if entry.get("client") is not None:
        def request_key(image_path, text):
            images = [image_path] + ([right_image_path] if right_image_path else [])
            payload = entry["client"].request(task_prompt(task, text), images,
                                                max_tokens=32)
            return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode(),
                                   digest_size=16).digest()
        return request_key
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
            digest.update(str(tensor.dtype).encode())
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=next(entry["model"].parameters()).dtype)
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.digest()

    def key(image_path, text):
        if name.startswith("qwen3-vl"):
            return serialize(modern_inputs(task, entry, image_path, text,
                                           right_image_path=right_image_path))
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


def load_paper_model(task, name, device, *, assets_dir=".", api_log=None):
    task, name = task.lower(), name.lower()
    if name in {"gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gemini-2.5-flash"}:
        from vision_client import VisionClient
        gemini = name == "gemini-2.5-flash"
        key_env = "GEMINI_API_KEY" if gemini else "OPENAI_API_KEY"
        client = VisionClient(
            "gemini-2.5-flash" if gemini else "gpt-4o-mini-2024-07-18",
            os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            if gemini else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ[key_env], provider="gemini" if gemini else "openai", log=api_log)
        return {"name": name, "client": client, "stochastic": True}
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
    if name in {"qwen3-vl-8b", "qwen3-vl-32b", "qwen3-vl-8b-instruct", "qwen3-vl-32b-instruct"}:
        from transformers import Qwen3VLForConditionalGeneration
        scale = "32B" if "32b" in name else "8B"
        checkpoint = f"Qwen/Qwen3-VL-{scale}-Instruct"
        return {"name": name, "processor": AutoProcessor.from_pretrained(checkpoint),
                "model": Qwen3VLForConditionalGeneration.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16,
                    device_map="auto" if str(device).startswith("cuda") else {"": "cpu"}).eval()}
    if task == "itm" and name == "vilt":
        checkpoint = "dandelin/vilt-b32-finetuned-coco"
        return {"name": name, "processor": ViltProcessor.from_pretrained(checkpoint),
                "model": ViltForImageAndTextRetrieval.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16).to(device).eval()}
    if task == "itm" and name == "blip":
        checkpoint = "Salesforce/blip-itm-base-coco"
        return {"name": name, "processor": BlipProcessor.from_pretrained(checkpoint),
                "model": BlipForImageTextRetrieval.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16).to(device).eval()}
    if task in {"vr", "nlvr"} and name == "vilt":
        checkpoint = "dandelin/vilt-b32-finetuned-nlvr2"
        return {"name": name, "processor": ViltProcessor.from_pretrained(checkpoint),
                "model": ViltForImagesAndTextClassification.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16).to(device).eval()}
    if task == "vqa" and name == "paligemma":
        checkpoint = "google/paligemma-3b-ft-vqav2-224"
        return {"name": name, "processor": AutoProcessor.from_pretrained(checkpoint),
                "model": PaliGemmaForConditionalGeneration.from_pretrained(
                    checkpoint, torch_dtype=torch.bfloat16).to(device).eval()}
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
                    "model": model.to(device=device, dtype=torch.bfloat16).eval()}
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
                "label2ans": label2ans,
                "model": model.to(device=device, dtype=torch.bfloat16).eval()}
    raise ValueError(f"unsupported task/model combination: {task}/{name}")


def make_target(task, entry, device, *, right_image_path=None, with_scores=False):
    if entry.get("client") is not None:
        def api_target(image_path, text):
            images = [image_path] + ([right_image_path] if right_image_path else [])
            return entry["client"].complete(task_prompt(task, text), images, max_tokens=32)
        return api_target
    import torch
    import torchvision.transforms as T
    from PIL import Image

    name = entry["name"].lower()
    task = task.lower()
    model = entry["model"]
    model.eval()
    model_dtype = next(model.parameters()).dtype

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
        if name.startswith("qwen3-vl"):
            inputs = modern_inputs(task, entry, image_path, text,
                                   right_image_path=right_image_path).to(device)
            length = inputs["input_ids"].shape[-1]
            generated = model.generate(**inputs, max_new_tokens=32,
                                       do_sample=False, num_beams=1)
            return entry["processor"].decode(generated[0][length:],
                                              skip_special_tokens=True).strip()
        image = Image.open(image_path).convert("RGB")
        if task == "itm":
            inputs = entry["processor"](image, text, return_tensors="pt").to(
                device, dtype=model_dtype)
            outputs = model(**inputs)
            if name == "vilt":
                logit = outputs.logits.reshape(-1)[0].float()
                if with_scores:
                    target.scores = {0: float(torch.nn.functional.logsigmoid(-logit)),
                                     1: float(torch.nn.functional.logsigmoid(logit))}
                return int(logit.item() > 0)
            if name == "blip":
                if with_scores:
                    target.scores = dict(enumerate(outputs.itm_score[0].float().log_softmax(-1).tolist()))
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
                    pixel_values=inputs.pixel_values.unsqueeze(0).to(device, dtype=model_dtype))
                if with_scores:
                    target.scores = dict(enumerate(outputs.logits[0].float().log_softmax(-1).tolist()))
                return int(outputs.logits.argmax(dim=-1).item())
            if name == "beit3":
                tokens = entry["tokenizer"](
                    text, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=40)
                logits = model(
                    beit3_vr_transform(image).unsqueeze(0).to(device, dtype=model_dtype),
                    beit3_vr_transform(right).unsqueeze(0).to(device, dtype=model_dtype),
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].eq(0).to(device))
                if with_scores:
                    target.scores = dict(enumerate(logits[0].float().log_softmax(-1).tolist()))
                return int(logits.argmax(dim=-1).item())

        if task == "vqa":
            if name == "paligemma":
                prompt = f"<image> question: {text} short answer:"
                inputs = entry["processor"](
                    text=prompt, images=image, return_tensors="pt").to(device, dtype=model_dtype)
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
                    beit3_vqa_transform(image).unsqueeze(0).to(device, dtype=model_dtype),
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].eq(0).to(device))
                index = int(logits.argmax(dim=-1).item())
                if with_scores:
                    target.scores = dict(zip(entry["label2ans"], logits[0].float().log_softmax(-1).tolist()))
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


def task_prompt(task, text):
    if task == "itm":
        return f'Does the image match this caption? Reply only "match" or "no match".\nCaption: {text}'
    if task in {"vr", "nlvr"}:
        return f'Is this statement true for the two images? Reply only "true" or "false".\nStatement: {text}'
    return f"Answer the question with a short answer.\nQuestion: {text}"


def normalize_binary(output):
    if isinstance(output, (int, bool)) and output in {0, 1}:
        return int(output)
    label = re.sub(r"[.\s]+$", "", str(output).strip().lower())
    return {"match": 1, "no match": 0, "true": 1, "false": 0,
            "1": 1, "0": 0}.get(label)


def modern_inputs(task, entry, image_path, text, *, right_image_path=None):
    from PIL import Image
    paths = [image_path] + ([right_image_path] if right_image_path else [])
    content = []
    for path in paths:
        with Image.open(path) as image:
            content.append({"type": "image", "image": image.convert("RGB").copy()})
    content.append({"type": "text", "text": task_prompt(task, text)})
    return entry["processor"].apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")


def make_score(task, entry, device, *, right_image_path=None, normalize=lambda x: x):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image

    model, name = entry["model"], entry["name"]
    model_dtype = next(model.parameters()).dtype
    vr_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    vqa_transform = T.Compose([
        T.Resize((480, 480), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @torch.no_grad()
    def score(image_path, text, answer):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if name == "paligemma":
            prompt = f"<image> question: {text} short answer:"
            inputs = entry["processor"](text=prompt, images=image, suffix=str(answer),
                                          return_tensors="pt").to(device, dtype=model_dtype)
            labels = inputs["labels"][..., 1:]
            logits = model(**inputs).logits[..., :-1, :].float()
            mask = labels.ne(-100)
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
            return float(gathered[mask].mean())
        if name == "beit3":
            vr = task in {"vr", "nlvr"}
            tokens = entry["tokenizer"](text, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=40 if vr else 64)
            transform = vr_transform if vr else vqa_transform
            args = [transform(image).unsqueeze(0).to(device, dtype=model_dtype)]
            if vr:
                with Image.open(right_image_path) as right:
                    args.append(transform(right.convert("RGB")).unsqueeze(0).to(
                        device, dtype=model_dtype))
            args.extend([tokens["input_ids"].to(device), tokens["attention_mask"].eq(0).to(device)])
            logits = model(*args)[0].float()
            if vr:
                return float(logits[int(answer)])
            matches = [i for i, value in enumerate(entry["label2ans"])
                       if normalize(value) == normalize(answer)]
            return max((float(logits[i]) for i in matches), default=float("-inf"))
        if task in {"vr", "nlvr"}:
            with Image.open(right_image_path) as right:
                inputs = entry["processor"]([image, right.convert("RGB")], text,
                    return_tensors="pt", truncation=True, max_length=40).to(device)
            logits = model(input_ids=inputs.input_ids,
                           pixel_values=inputs.pixel_values.unsqueeze(0).to(dtype=model_dtype)).logits[0]
        else:
            inputs = entry["processor"](image, text, return_tensors="pt").to(device, dtype=model_dtype)
            outputs = model(**inputs)
            logits = outputs.itm_score[0] if name == "blip" else outputs.logits.reshape(-1)
            if logits.numel() == 1:
                logits = torch.stack((torch.zeros_like(logits[0]), logits[0]))
        return float(logits[int(answer)])

    return score
