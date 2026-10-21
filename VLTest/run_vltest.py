
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from method import _MISSING, discover
from proxies import (
    make_relation_proxies,
    nli_scorer,
    openclip_scorers,
)
from targets import load_paper_model, make_preprocessed_key, make_score, make_target, normalize_binary
from text_variants import generate_text_pool, pegasus_rewrites, roberta_mask_predictor
from verification import VLMOracle, interaction_analysis, retest_arm_pattern, retest_joint_pattern
from visual import generate_visual_pool
from vision_client import VisionClient


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["itm", "vr", "vqa"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["VLTest", "VLTest-V", "VLTest-T",
                        "SINVAD-VQ", "TextFooler", "EXT", "MetaRA-c"], default="VLTest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=Path("."))
    parser.add_argument("--vqgan-config", type=Path,
                        default=Path(__file__).parent / "configs" / "gumbel_f8.yaml")
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--joint-budget", type=int)
    parser.add_argument("--image-arm-cap", type=int)
    parser.add_argument("--text-arm-cap", type=int)
    parser.add_argument("--policy", choices=["RR", "RND", "CRND", "SR", "SF", "SAF", "JF", "MIX"],
                        default="SF")
    parser.add_argument("--rank-range", type=int, default=8191)
    parser.add_argument("--kappa", type=int, default=15)
    parser.add_argument("--beta", type=int, default=4000)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--disable-proxies", action="store_true")
    parser.add_argument("--nli-threshold", type=float, default=0.80)
    parser.add_argument("--image-threshold", type=float, default=0.90)
    parser.add_argument("--alignment-ratio", type=float, default=0.95)
    parser.add_argument("--negative-margin", type=float, default=0.05)
    parser.add_argument("--verifier-base-url", default=os.environ.get(
        "VLTEST_VERIFIER_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--verifier-model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    parser.add_argument("--verifier-key-env", default="VLTEST_VERIFIER_API_KEY")
    parser.add_argument("--interaction-analysis", action="store_true")
    parser.add_argument("--relation-manifest", type=Path)
    args = parser.parse_args(argv)
    args.image_arm_cap = args.budget // 2 if args.image_arm_cap is None else args.image_arm_cap
    args.text_arm_cap = args.budget // 2 if args.text_arm_cap is None else args.text_arm_cap
    args.joint_budget = args.budget if args.joint_budget is None else args.joint_budget
    if args.method in {"VLTest-V", "SINVAD-VQ"}:
        args.image_arm_cap, args.text_arm_cap, args.joint_budget = args.budget, 0, 0
        args.policy = "SF"
    elif args.method in {"VLTest-T", "TextFooler"}:
        args.image_arm_cap, args.text_arm_cap, args.joint_budget = 0, args.budget, 0
        args.policy = "SF"
    elif args.method == "EXT":
        args.policy = "SF"
    elif args.method == "MetaRA-c":
        if args.task != "vqa" or args.relation_manifest is None:
            parser.error("MetaRA-c requires VQA and --relation-manifest with the specified transformations and pairs")
        args.disable_proxies = True
        args.policy = "RR"
    args.disable_proxies = args.disable_proxies or args.policy == "RND"
    if (args.interaction_analysis or args.method in {"SINVAD-VQ", "TextFooler", "EXT"}) and args.model.lower() not in {"blip", "vilt", "beit3", "paligemma"}:
        parser.error("this configuration requires a task-specific model with an output-score adapter")
    return args


def resolve_path(value, manifest):
    path = Path(value)
    return str(path if path.is_absolute() else manifest.parent / path)


def seed_answers(record, task, normalize):
    if task in {"itm", "vr"}:
        return {int(record["label"])}
    if "reference_answers" in record:
        return {normalize(item) for item in record["reference_answers"]}
    counts = Counter(normalize(item["answer"] if isinstance(item, dict) else item)
                     for item in record["answers"])
    return {answer for answer, count in counts.items() if count >= 4}


def main():
    args = arguments()
    from importlib import import_module

    import spacy
    import torch
    from omegaconf import OmegaConf
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from vqa_normalizer_exact import VQANormalizerExact

    if args.budget < 0 or args.joint_budget < 0:
        raise ValueError("budgets must be nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.vqgan_config)
    first_stage = config.model.params.get("first_stage_config", config.model)
    if first_stage["target"] != "taming.models.vqgan.GumbelVQ":
        raise ValueError("the paper configuration requires the Gumbel f8 tokenizer")
    if (first_stage.params.n_embed != 8192 or
            len(first_stage.params.ddconfig.ch_mult) != 4):
        raise ValueError("the paper tokenizer requires 8192 entries and downsampling factor 8")
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
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
        image_similarity, alignment = openclip_scorers(
            clip_model.eval(), clip_preprocess, open_clip.get_tokenizer("ViT-bigG-14"), device)

    normalize = (VQANormalizerExact().normalize if args.task == "vqa" else normalize_binary)
    with args.manifest.open(encoding="utf-8") as handle:
        seeds = [json.loads(line) for line in handle if line.strip()]
    seed_ids = [str(record.get("seed_id", index))
                for index, record in enumerate(seeds)]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed IDs must be unique")
    source_paths = [resolve_path(record["image"], args.manifest) for record in seeds]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("seed manifest must use distinct source images")
    relations = {}
    if args.relation_manifest is not None:
        with args.relation_manifest.open(encoding="utf-8") as handle:
            rows = (json.loads(line) for line in handle if line.strip())
            relations = {str(row["seed_id"]): row for row in rows}

    calls_path = args.output / "discovery_calls.jsonl"
    candidates_path = args.output / "candidates.jsonl"
    blind_path = args.output / "annotation_reference_blind.jsonl"
    summary_path = args.output / "seed_summary.jsonl"
    screening_path = args.output / "screening.jsonl"
    run_id = str(args.output.resolve())
    with ExitStack() as stack:
        api_log = stack.enter_context((args.output / "api_calls.jsonl").open("w", encoding="utf-8"))
        verifier_log = stack.enter_context((args.output / "verifier_calls.jsonl").open("w", encoding="utf-8"))
        verification_calls = stack.enter_context((args.output / "verification_calls.jsonl").open("w", encoding="utf-8"))
        outcomes = stack.enter_context((args.output / "oracle_outcomes.jsonl").open("w", encoding="utf-8"))
        evaluation_calls = stack.enter_context((args.output / "evaluation_calls.jsonl").open("w", encoding="utf-8")) if args.interaction_analysis else None
        verifier = VisionClient(args.verifier_model, args.verifier_base_url,
                                os.environ.get(args.verifier_key_env, ""), log=verifier_log)
        verifier.role = "verifier"
        entry = load_paper_model(args.task, args.model, device,
                                 assets_dir=args.assets_dir, api_log=api_log)
        calls = stack.enter_context(calls_path.open("w", encoding="utf-8"))
        candidates = stack.enter_context(candidates_path.open("w", encoding="utf-8"))
        blind = stack.enter_context(blind_path.open("w", encoding="utf-8"))
        summaries = stack.enter_context(summary_path.open("w", encoding="utf-8"))
        screening = stack.enter_context(screening_path.open("w", encoding="utf-8"))
        for seed_number, record in enumerate(seeds):
            verifier_start = verifier.calls
            random.seed(args.seed ^ seed_number)
            torch.manual_seed(args.seed ^ seed_number)
            seed_id = str(record.get("seed_id", seed_number))
            raw_image = resolve_path(record["image"], args.manifest)
            right_image = (resolve_path(record["right_image"], args.manifest)
                           if args.task == "vr" else None)
            text = record["text"]
            expected = seed_answers(record, args.task, normalize)
            if not expected:
                continue
            target = make_target(args.task, entry, device,
                                 right_image_path=right_image,
                                 with_scores=args.method in {"SINVAD-VQ", "TextFooler", "EXT"})
            input_key = make_preprocessed_key(args.task, entry,
                                              right_image_path=right_image)
            if entry.get("client") is not None:
                entry["client"].role = "raw_seed"
            seed_output = target(raw_image, text)
            oracle = VLMOracle(verifier, task=args.task, seed=(raw_image, text),
                               answers=expected, right_image=right_image)
            reconstruction_output = _MISSING

            def admit(path):
                nonlocal reconstruction_output
                if oracle.answer_check((path, text)) != "preserved":
                    return False
                if entry.get("client") is not None:
                    entry["client"].role = "seed"
                reconstruction_output = (seed_output if input_key(path, text) == input_key(raw_image, text)
                                         else target(path, text))
                return normalize(reconstruction_output) in expected

            pool = generate_visual_pool(
                vqgan, raw_image, str(args.output / "images" / seed_id),
                top_k=args.kappa, rank_range=args.rank_range,
                model_input_key=partial(input_key, text=text), admit=admit,
                max_variants=0 if args.method in {"VLTest-T", "TextFooler", "SINVAD-VQ", "EXT", "MetaRA-c"} else None)
            reconstruction_path = pool.reconstruction_path
            if entry.get("client") is not None:
                entry["client"].role = "discovery"
            images, texts = [], []
            proxies = None
            if reconstruction_output is not _MISSING and normalize(reconstruction_output) in expected:
                images = pool.variants
                if args.method in {"VLTest", "VLTest-T"}:
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
                        negative_margin=args.negative_margin,
                        reconstruction=reconstruction_path)
            image_search = text_search = score_target = pair_order = None
            if args.method in {"SINVAD-VQ", "TextFooler", "EXT"}:
                from baselines import SINVADSearch, TextFoolerSearch
                score_target = make_score(args.task, entry, device,
                                          right_image_path=right_image, normalize=normalize)
                if args.method in {"SINVAD-VQ", "EXT"}:
                    image_search = SINVADSearch(vqgan, raw_image,
                        args.output / "images" / seed_id / "sinvad", random_seed=args.seed ^ seed_number)
                if args.method in {"TextFooler", "EXT"}:
                    text_search = TextFoolerSearch(text)
            elif args.method == "MetaRA-c":
                relation = relations[seed_id]
                images = [resolve_path(path, args.relation_manifest) for path in relation["images"]]
                texts = relation["texts"]

                def pair_order(image_pool, text_pool):
                    for image_index, text_index in relation["pairs"]:
                        image, question = images[image_index], texts[text_index]
                        if image in image_pool and question in text_pool:
                            yield image, question

            result = discover(
                seed_image=raw_image, seed_text=text,
                reconstruction=reconstruction_path,
                acceptable_answers=expected, image_variants=images,
                text_variants=texts, target=target, normalize=normalize,
                answer_check=oracle.answer_check,
                budget=args.budget, image_arm_cap=args.image_arm_cap,
                text_arm_cap=args.text_arm_cap, joint_budget=args.joint_budget, policy=args.policy,
                proxies=proxies, use_proxies=not args.disable_proxies,
                random_seed=args.seed ^ seed_number, input_key=input_key,
                seed_output=seed_output, reconstruction_output=reconstruction_output,
                image_search=image_search, text_search=text_search, score_target=score_target,
                pair_order=pair_order)
            reconstruction = result.reconstruction_observation
            summary = {
                "seed_id": seed_id, "task": args.task, "model": args.model,
                "admitted": result.admitted,
                "discovery_calls": result.discovery_calls,
                "single_calls": result.single_calls, "joint_calls": result.joint_calls,
                "reference_calls": len(result.log) - result.discovery_calls,
                "single_budget": args.budget, "joint_budget": args.joint_budget,
                "reconstruction_answer": oracle.answer_check((reconstruction_path, text)),
                "reused_observations": result.reused,
                "reconstruction_sensitive": (
                    reconstruction.normalized != normalize(seed_output)
                    if reconstruction is not None else None),
                "image_candidates": len(result.image_candidates),
                "text_candidates": len(result.text_candidates),
                "joint_candidates": len(result.joint_candidates),
                "output_changes": sum(
                    item.normalized != reconstruction.normalized
                    for item in result.log
                    if item.purpose in {"image_arm", "text_arm", "joint"}
                    and reconstruction is not None),
            }
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
                    "request_type": observation.request_type,
                    "purpose": observation.purpose,
                    "image": observation.cell[0], "text": observation.cell[1],
                    "output": observation.output,
                    "normalized": observation.normalized,
                }, ensure_ascii=False) + "\n")
            if not result.admitted:
                summary["C_verifier"] = verifier.calls - verifier_start
                summary["C_ver"] = 0
                summaries.write(json.dumps(summary, ensure_ascii=False) + "\n")
                continue
            tuples = [
                ("image_arm", (reconstruction_path, text), cell) for cell in result.image_candidates
            ] + [
                ("text_arm", (reconstruction_path, text), cell) for cell in result.text_candidates
            ] + [
                ("joint", (reconstruction_path, text), candidate.joint)
                for candidate in result.joint_candidates
            ]
            joint_by_cell = {candidate.joint: candidate
                             for candidate in result.joint_candidates}
            observations = {input_key(*item.cell): item for item in result.log
                            if item.request_type == "predict"}
            retest_outputs, retest_positions = {}, {}

            def retest_target(image, question):
                cell = (image, question)
                identity = input_key(*cell)
                if identity not in retest_outputs:
                    if entry.get("client") is not None:
                        entry["client"].role = "verification"
                    retest_outputs[identity] = []
                    for repeat in range(5):
                        output = target(*cell)
                        retest_outputs[identity].append(output)
                        verification_calls.write(json.dumps({
                            "seed_id": seed_id, "role": "verification", "repeat": repeat,
                            "image": image, "text": question, "output": output,
                        }, ensure_ascii=False) + "\n")
                    if entry.get("client") is not None:
                        entry["client"].role = "discovery"
                position = retest_positions.get(identity, 0)
                retest_positions[identity] = (position + 1) % 5
                return retest_outputs[identity][position]

            base_score = (make_score(args.task, entry, device, right_image_path=right_image,
                                     normalize=normalize) if args.interaction_analysis else None)
            score_cache = {}

            def score(image, question, answer):
                identity = (input_key(image, question), answer)
                if identity not in score_cache:
                    value = base_score(image, question, answer)
                    score_cache[identity] = value
                    evaluation_calls.write(json.dumps({
                        "seed_id": seed_id, "role": "evaluation", "request_type": "score",
                        "image": image, "text": question, "answer": answer, "score": value,
                    }, ensure_ascii=False) + "\n")
                return score_cache[identity]

            confirmed_counts = Counter()
            verified_count = 0
            for index, (kind, seed_cell, cell) in enumerate(tuples):
                candidate_id = f"{run_id}:{seed_id}:{index}"
                cells = [seed_cell, cell]
                completion_call = None
                if kind == "joint":
                    match = joint_by_cell[cell]
                    cells = [match.seed, match.image_arm, match.text_arm,
                             match.joint]
                    completion_call = match.completion_call
                verdicts = [oracle.judge(pair, observations[input_key(*pair)].output)
                            for pair in cells[1:]]
                confirmed = verdicts[-1].failure
                verified = (kind == "joint" and confirmed and
                            all(verdict.correct for verdict in verdicts[:-1]))
                stable = None
                if entry.get("stochastic"):
                    kwargs = {"judge": oracle.output_check, "input_key": input_key}
                    if kind == "joint":
                        stability = retest_joint_pattern(match, result.log, retest_target,
                                                         normalize, expected, **kwargs)
                    else:
                        stability = retest_arm_pattern(seed_cell, cell, result.log,
                                                       retest_target, normalize, expected, **kwargs)
                    stable = stability.stable
                    confirmed = (confirmed and stability.compatible_votes[seed_cell] >= 3
                                 and stability.incorrect_votes[cell] >= 3)
                    verified = verified and stable
                outcome = {
                    "candidate_id": candidate_id, "seed_id": seed_id, "kind": kind,
                    "confirmed_failure": confirmed, "verified_joint_failure": verified,
                    "reproducible": stable,
                    "cells": [{"image": pair[0], "text": pair[1],
                               "answer": verdict.answer, "output": verdict.output,
                               "compatible": observations[input_key(*pair)].normalized in expected}
                              for pair, verdict in zip(cells[1:], verdicts)],
                }
                confirmed_counts[kind] += int(confirmed)
                verified_count += int(verified)
                if verified and base_score is not None:
                    outcome["interaction"] = interaction_analysis(
                        cells, expected, observations[input_key(*cell)].normalized, score)
                outcomes.write(json.dumps(outcome, ensure_ascii=False) + "\n")
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
            summary.update({
                "C_ver": 5 * len(retest_outputs), "C_verifier": verifier.calls - verifier_start,
                "C_eval": len(score_cache), "confirmed_image": confirmed_counts["image_arm"],
                "confirmed_text": confirmed_counts["text_arm"],
                "confirmed_joint": confirmed_counts["joint"], "verified_joint": verified_count,
                "CF@1k": 1000 * (confirmed_counts["image_arm"] + confirmed_counts["text_arm"]) / args.budget if args.budget else None,
                "VJF@1k": 1000 * verified_count / args.joint_budget if args.joint_budget else None,
            })
            summaries.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
