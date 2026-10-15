
from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence


def pegasus_rewrites(text, tokenizer, model, *, gamma: int, device,
                     max_rounds: int = 5) -> list[str]:
    import torch

    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    collected, seen = [], {text.casefold()}
    with torch.no_grad():
        for _ in range(max_rounds):
            if len(collected) >= gamma:
                break
            outputs = model.generate(
                **encoded, do_sample=True, temperature=0.8, top_p=0.9,
                num_return_sequences=gamma - len(collected), max_new_tokens=64,
                num_beams=1)
            for output in outputs:
                candidate = tokenizer.decode(output, skip_special_tokens=True).strip()
                if candidate and candidate.casefold() not in seen:
                    seen.add(candidate.casefold())
                    collected.append(candidate)
    return collected


def roberta_mask_predictor(tokenizer, model, device) -> Callable[[str, int], list[str]]:
    import torch

    def predict(masked_sentence: str, count: int) -> list[str]:
        encoded = tokenizer(masked_sentence, return_tensors="pt", truncation=True)
        locations = (encoded.input_ids[0] == tokenizer.mask_token_id).nonzero()
        if len(locations) != 1:
            return []
        index = int(locations[0, 0])
        with torch.no_grad():
            logits = model(**encoded.to(device)).logits[0, index]
        ids = logits.topk(min(count, logits.numel())).indices.tolist()
        return [tokenizer.decode([token_id]).strip() for token_id in ids]

    return predict


def generate_text_pool(
    seed_text: str,
    rewrites: Sequence[str],
    nlp,
    mask_token: str,
    predict_mask: Callable[[str, int], Sequence[str]],
    sentence_encoder,
    *,
    beta: int,
    tau: int,
    delta: int,
    random_seed: int,
) -> list[str]:
    import numpy as np

    if beta < 0 or tau < 1 or delta < 0:
        raise ValueError("invalid text generation parameters")
    if not rewrites or beta == 0:
        return []
    rng = random.Random(random_seed)
    groups: list[list[str]] = []

    def replace_spans(text: str, tokens, replacements: dict[int, str]) -> str:
        for token in sorted(tokens, key=lambda tok: tok.idx, reverse=True):
            if token.i in replacements:
                text = (text[:token.idx] + replacements[token.i] +
                        text[token.idx + len(token.text):])
        return text

    for rewrite in rewrites:
        doc = nlp(rewrite)
        eligible = [tok for tok in doc if tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                    and tok.is_alpha and not tok.ent_type_ and not tok.like_num]
        if len(eligible) > delta:
            eligible = rng.sample(eligible, delta)
        options = []
        for token in eligible:
            masked = replace_spans(rewrite, [token], {token.i: mask_token})
            choices = [token.text]
            for proposal in predict_mask(masked, tau * 10):
                if not proposal.isalpha() or proposal.casefold() == token.text.casefold():
                    continue
                parsed = nlp(proposal)
                if len(parsed) != 1 or parsed[0].pos_ != token.pos_:
                    continue
                if proposal not in choices:
                    choices.append(proposal)
                if len(choices) - 1 >= tau:
                    break
            options.append(choices)

        candidates = [rewrite]
        if options:
            for combination in itertools.product(*options):
                replacements = {token.i: value for token, value in zip(eligible, combination)
                                if value != token.text}
                if replacements:
                    candidates.append(replace_spans(rewrite, eligible, replacements))
        candidates = list(dict.fromkeys(candidates))
        embeddings = sentence_encoder.encode([seed_text] + candidates,
                                              normalize_embeddings=True)
        similarity = np.asarray(embeddings[1:]) @ np.asarray(embeddings[0])
        ranked = sorted(range(len(candidates)), key=lambda i: (-float(similarity[i]), i))
        groups.append([candidates[i] for i in ranked])

    group_count = len(groups)
    quotas = [beta // group_count + (index < beta % group_count)
              for index in range(group_count)]
    pool, seen = [], {seed_text}
    rank = 0
    while len(pool) < beta and any(rank < min(len(group), quotas[index])
                                   for index, group in enumerate(groups)):
        for index, group in enumerate(groups):
            if rank < min(len(group), quotas[index]) and group[rank] not in seen:
                pool.append(group[rank])
                seen.add(group[rank])
                if len(pool) == beta:
                    break
        rank += 1
    return pool
