
from __future__ import annotations

import argparse
import json
import random
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from method import discover
from proxies import (
    clip_image_scorer,
    make_relation_proxies,
    nli_scorer,
    siglip_pair_scorer,
)
from targets import load_paper_model, make_preprocessed_key, make_target
from text_variants import generate_text_pool, pegasus_rewrites, roberta_mask_predictor
from visual import generate_visual_pool


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["itm", "vr", "vqa"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=Path("."))
    parser.add_argument("--vqgan-config", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--image-arm-cap", type=int, required=True)
    parser.add_argument("--text-arm-cap", type=int, required=True)
    parser.add_argument("--policy", choices=["RR", "RND", "SAF", "JF", "MIX"],
                        default="RR")
    parser.add_argument("--rank-range", type=int, required=True)
    parser.add_argument("--kappa", type=int, default=15)
    parser.add_argument("--beta", type=int, default=1000)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--disable-proxies", action="store_true")
    parser.add_argument("--nli-threshold", type=float)
    parser.add_argument("--image-threshold", type=float)
    parser.add_argument("--alignment-ratio", type=float)
    parser.add_argument("--negative-margin", type=float)
    return parser.parse_args()


def resolve_path(value, manifest):
    path = Path(value)
    return str(path if path.is_absolute() else manifest.parent / path)


def seed_answers(record, task, normalize):
    if task in {"itm", "vr"}:
        return {int(record["label"])}
    answers = record["answers"]
    return {normalize(item["answer"] if isinstance(item, dict) else item)
            for item in answers}


def main():
    args = arguments()
    from importlib import import_module

    import spacy
    import torch
    from omegaconf import OmegaConf
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
        CLIPModel,
        CLIPProcessor,
    )
    from vqa_normalizer_exact import VQANormalizerExact

    if not args.disable_proxies and any(value is None for value in (
            args.nli_threshold, args.image_threshold,
            args.alignment_ratio, args.negative_margin)):
        raise ValueError("proxy thresholds must be set from the pilot annotation")
    if args.budget < 2:
        raise ValueError("budget must cover the seed and reconstruction control")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.vqgan_config)
    first_stage = config.model.params.first_stage_config
    first_stage.params.ckpt_path = str(args.vqgan_checkpoint)
    module_name, class_name = first_stage["target"].rsplit(".", 1)
    vqgan = getattr(import_module(module_name), class_name)(
        **first_stage.get("params", {})).to(device).eval()
    checkpoint = torch.load(args.vqgan_checkpoint, map_location="cpu")
    vqgan.load_state_dict(checkpoint["state_dict"], strict=False)
    vqgan.to(device).eval()

    paraphrase_name = "tuner007/pegasus_paraphrase"
    paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_name)
    paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(
        paraphrase_name).to(device).eval()
    mask_name = "FacebookAI/roberta-large"
    mask_tokenizer = AutoTokenizer.from_pretrained(mask_name)
    mask_model = AutoModelForMaskedLM.from_pretrained(mask_name).to(device).eval()
    mask_predict = roberta_mask_predictor(mask_tokenizer, mask_model, device)
    language = spacy.load("en_core_web_sm", disable=["parser"])
    sentence_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=str(device))

    if not args.disable_proxies:
        nli_name = "FacebookAI/roberta-large-mnli"
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_name).to(device).eval()
        entailment = nli_scorer(nli_tokenizer, nli_model, device)
        clip_name = "openai/clip-vit-base-patch32"
        clip_processor = CLIPProcessor.from_pretrained(clip_name)
        clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        image_similarity = clip_image_scorer(clip_processor, clip_model, device)
        siglip_name = "google/siglip-base-patch16-224"
        siglip_processor = AutoProcessor.from_pretrained(siglip_name)
        siglip_model = AutoModel.from_pretrained(siglip_name).to(device).eval()
        alignment = siglip_pair_scorer(siglip_processor, siglip_model, device)

    normalize = (VQANormalizerExact().normalize if args.task == "vqa" else int)
    entry = load_paper_model(args.task, args.model, device,
                             assets_dir=args.assets_dir)
    with args.manifest.open(encoding="utf-8") as handle:
        seeds = [json.loads(line) for line in handle if line.strip()]
    seed_ids = [str(record.get("seed_id", index))
                for index, record in enumerate(seeds)]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed IDs must be unique")
    source_paths = [resolve_path(record["image"], args.manifest) for record in seeds]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("seed manifest must use distinct source images")

    calls_path = args.output / "discovery_calls.jsonl"
    candidates_path = args.output / "candidates.jsonl"
    blind_path = args.output / "annotation_reference_blind.jsonl"
    summary_path = args.output / "seed_summary.jsonl"
    screening_path = args.output / "screening.jsonl"
    run_id = str(args.output.resolve())
    with ExitStack() as stack:
        calls = stack.enter_context(calls_path.open("w", encoding="utf-8"))
        candidates = stack.enter_context(candidates_path.open("w", encoding="utf-8"))
        blind = stack.enter_context(blind_path.open("w", encoding="utf-8"))
        summaries = stack.enter_context(summary_path.open("w", encoding="utf-8"))
        screening = stack.enter_context(screening_path.open("w", encoding="utf-8"))
        for seed_number, record in enumerate(seeds):
            seed_id = str(record.get("seed_id", seed_number))
            raw_image = resolve_path(record["image"], args.manifest)
            right_image = (resolve_path(record["right_image"], args.manifest)
                           if args.task == "vr" else None)
            text = record["text"]
            expected = seed_answers(record, args.task, normalize)
            if not expected:
                continue
            target = make_target(args.task, entry, device,
                                 right_image_path=right_image)
            input_key = make_preprocessed_key(args.task, entry,
                                              right_image_path=right_image)
            seed_output = target(raw_image, text)
            admitted = normalize(seed_output) in expected
            reconstruction_path = raw_image
            images, texts = [], []
            proxies = None
            if admitted:
                pool = generate_visual_pool(
                    vqgan, raw_image, str(args.output / "images" / seed_id),
                    top_k=args.kappa, rank_range=args.rank_range,
                    max_variants=min(args.rank_range, args.budget),
                    model_input_key=partial(input_key, text=text))
                reconstruction_path = pool.reconstruction_path
                images = pool.variants
                rewrites = pegasus_rewrites(
                    text, paraphrase_tokenizer, paraphrase_model,
                    gamma=args.gamma, device=device)
                texts = generate_text_pool(
                    text, rewrites, language, mask_tokenizer.mask_token,
                    mask_predict, sentence_encoder, beta=args.beta,
                    tau=args.tau, delta=args.delta,
                    random_seed=args.seed ^ seed_number)
                if not args.disable_proxies:
                    proxies = make_relation_proxies(
                        task=args.task, seed_image=raw_image, seed_text=text,
                        seed_label=record.get("label"),
                        entailment_probability=entailment,
                        image_similarity=image_similarity,
                        alignment_score=alignment,
                        nli_threshold=args.nli_threshold,
                        image_threshold=args.image_threshold,
                        alignment_ratio=args.alignment_ratio,
                        negative_margin=args.negative_margin)
            result = discover(
                seed_image=raw_image, seed_text=text,
                reconstruction=reconstruction_path,
                acceptable_answers=expected, image_variants=images,
                text_variants=texts, target=target, normalize=normalize,
                budget=args.budget, image_arm_cap=args.image_arm_cap,
                text_arm_cap=args.text_arm_cap, policy=args.policy,
                proxies=proxies, use_proxies=not args.disable_proxies,
                random_seed=args.seed ^ seed_number, input_key=input_key,
                seed_output=seed_output)
            reconstruction = result.reconstruction_observation
            summaries.write(json.dumps({
                "seed_id": seed_id, "task": args.task, "model": args.model,
                "admitted": result.admitted,
                "discovery_calls": result.discovery_calls,
                "reused_observations": result.reused,
                "reconstruction_sensitive": (
                    reconstruction.normalized not in expected
                    if reconstruction is not None else None),
                "image_candidates": len(result.image_candidates),
                "text_candidates": len(result.text_candidates),
                "joint_candidates": len(result.joint_candidates),
                "output_changes": sum(
                    item.normalized != result.log[0].normalized
                    for item in result.log[1:]
                    if item.purpose != "reconstruction"),
            }, ensure_ascii=False) + "\n")
            for screening_index, screened in enumerate(result.screening):
                screening.write(json.dumps({
                    "seed_id": seed_id, "task": args.task, "model": args.model,
                    "policy": args.policy, "run_id": run_id,
                    "proxies_enabled": not args.disable_proxies,
                    "variant_id": f"{run_id}:{seed_id}:screen:{screening_index}",
                    **screened,
                }, ensure_ascii=False) + "\n")
            for observation in result.log:
                calls.write(json.dumps({
                    "seed_id": seed_id, "task": args.task, "model": args.model,
                    "call_index": observation.call_index,
                    "input_id": observation.input_id,
                    "purpose": observation.purpose,
                    "image": observation.cell[0], "text": observation.cell[1],
                    "output": observation.output,
                    "normalized": observation.normalized,
                }, ensure_ascii=False) + "\n")
            if not result.admitted:
                continue
            tuples = [
                ("image_arm", (raw_image, text), cell) for cell in result.image_candidates
            ] + [
                ("text_arm", (raw_image, text), cell) for cell in result.text_candidates
            ] + [
                ("joint", (raw_image, text), candidate.joint)
                for candidate in result.joint_candidates
            ]
            joint_by_cell = {candidate.joint: candidate
                             for candidate in result.joint_candidates}
            for index, (kind, seed_cell, cell) in enumerate(tuples):
                candidate_id = f"{run_id}:{seed_id}:{index}"
                cells = [seed_cell, cell]
                completion_call = None
                if kind == "joint":
                    match = joint_by_cell[cell]
                    cells = [match.seed, match.image_arm, match.text_arm,
                             match.joint]
                    completion_call = match.completion_call
                candidates.write(json.dumps({
                    "candidate_id": candidate_id, "seed_id": seed_id,
                    "task": args.task, "model": args.model,
                    "policy": args.policy, "run_id": run_id,
                    "proxies_enabled": not args.disable_proxies,
                    "kind": kind, "cells": cells,
                    "completion_call": completion_call,
                }, ensure_ascii=False) + "\n")
                for role, pair in enumerate(cells):
                    blind.write(json.dumps({
                        "candidate_id": candidate_id, "seed_id": seed_id,
                        "cell_role": role, "input_id": input_key(*pair).hex(),
                        "image": pair[0], "text": pair[1],
                        "seed_answers": sorted(expected),
                        "right_image": right_image,
                    }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
