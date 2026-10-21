
from __future__ import annotations

from method import Proxies


def make_relation_proxies(
    *,
    task: str,
    seed_image,
    seed_text: str,
    seed_label,
    entailment_probability,
    image_similarity,
    alignment_score,
    nli_threshold: float,
    image_threshold: float,
    alignment_ratio: float,
    negative_margin: float,
    reconstruction=None,
) -> Proxies:
    task = task.lower()
    if task not in {"itm", "vr", "nlvr", "vqa"}:
        raise ValueError(f"unknown task: {task}")
    uses_alignment = task == "itm" or (task in {"vr", "nlvr"} and
                                            seed_label in {True, "True"})
    reference = seed_image if reconstruction is None else reconstruction
    baseline = alignment_score(reference, seed_text) if uses_alignment else None

    def text_ok(text, original):
        return min(entailment_probability(original, text),
                   entailment_probability(text, original)) >= nli_threshold

    def image_ok(image, original):
        return image_similarity(image, original) >= image_threshold

    def pair_ok(image, text):
        if not uses_alignment:
            return True
        score = alignment_score(image, text)
        if task == "itm" and seed_label in {False, "False"}:
            return score <= baseline + negative_margin
        return score >= alignment_ratio * baseline

    return Proxies(image=image_ok, text=text_ok, pair=pair_ok)


def nli_scorer(tokenizer, model, device):
    import torch

    label_map = {str(label).lower(): index
                 for index, label in model.config.id2label.items()}
    entailment_id = next((index for label, index in label_map.items()
                          if "entail" in label), None)
    if entailment_id is None:
        raise ValueError("NLI checkpoint has no entailment label")

    @torch.no_grad()
    def score(premise, hypothesis):
        inputs = tokenizer(premise, hypothesis, truncation=True,
                           return_tensors="pt").to(device)
        logits = model(**inputs).logits[0]
        return float(logits.softmax(-1)[entailment_id].item())

    return score


def openclip_scorers(model, preprocess, tokenizer, device):
    from functools import lru_cache

    import torch
    import torch.nn.functional as F
    from PIL import Image

    @lru_cache(maxsize=8192)
    @torch.no_grad()
    def image_embedding(path):
        with Image.open(path) as image:
            pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        return F.normalize(model.encode_image(pixels).float(), dim=-1).cpu()

    @lru_cache(maxsize=8192)
    @torch.no_grad()
    def text_embedding(text):
        tokens = tokenizer([text]).to(device)
        return F.normalize(model.encode_text(tokens).float(), dim=-1).cpu()

    def image_similarity(path, seed_path):
        return float((image_embedding(path) * image_embedding(seed_path)).sum())

    def alignment(path, text):
        return float((image_embedding(path) * text_embedding(text)).sum())

    return image_similarity, alignment


def clip_image_scorer(processor, model, device):
    from functools import lru_cache

    import torch
    import torch.nn.functional as F
    from PIL import Image

    @lru_cache(maxsize=8192)
    @torch.no_grad()
    def embedding(path):
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"),
                               return_tensors="pt").to(device)
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return F.normalize(features, dim=-1).cpu()

    def score(path, seed_path):
        return float((embedding(path) * embedding(seed_path)).sum().item())

    return score


def siglip_pair_scorer(processor, model, device):
    from functools import lru_cache

    import torch
    from PIL import Image

    @lru_cache(maxsize=65536)
    @torch.no_grad()
    def score(path, text):
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"), text=text,
                               padding="max_length", return_tensors="pt").to(device)
        output = model(**inputs)
        return float(output.logits_per_image[0, 0].item())

    return score
