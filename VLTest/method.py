
from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Cell = tuple[Any, str]
Pair = tuple[Any, str]
_MISSING = object()


@dataclass
class Observation:
    cell: Cell
    output: Any
    normalized: Any
    purpose: str
    call_index: int
    input_id: str
    request_type: str = "predict"
    scores: dict[Any, float] | None = None


@dataclass
class JointCandidate:
    seed: Cell
    image_arm: Cell
    text_arm: Cell
    joint: Cell
    completion_call: int


@dataclass
class DiscoveryResult:
    admitted: bool
    log: list[Observation] = field(default_factory=list)
    reconstruction_observation: Observation | None = None
    reused: int = 0
    image_candidates: list[Cell] = field(default_factory=list)
    text_candidates: list[Cell] = field(default_factory=list)
    joint_candidates: list[JointCandidate] = field(default_factory=list)
    screening: list[dict[str, Any]] = field(default_factory=list)

    @property
    def discovery_calls(self) -> int:
        return self.single_calls + self.joint_calls

    @property
    def single_calls(self) -> int:
        return sum(item.purpose in {"image_arm", "text_arm", "score"}
                   for item in self.log)

    @property
    def joint_calls(self) -> int:
        return sum(item.purpose == "joint" for item in self.log)


@dataclass
class Proxies:
    image: Callable[[Any, Any], bool] = lambda _x, _x0: True
    text: Callable[[str, str], bool] = lambda _t, _t0: True
    pair: Callable[[Any, str], bool] = lambda _x, _t: True


def _rr_pairs(images: Sequence[Any], texts: Sequence[str]) -> Iterable[Pair]:
    if not images or not texts:
        return
    image_longer = len(images) >= len(texts)
    longer, shorter = (images, texts) if image_longer else (texts, images)
    na, nb = len(longer), len(shorter)
    for k in range(na * nb):
        a = longer[k % na]
        b = shorter[((k % na) + k // na) % nb]
        yield (a, b) if image_longer else (b, a)


def _interleave(image_arms: Sequence[Cell], text_arms: Sequence[Cell]) -> Iterable[Cell]:
    for i in range(max(len(image_arms), len(text_arms))):
        if i < len(image_arms):
            yield image_arms[i]
        if i < len(text_arms):
            yield text_arms[i]


def discover(
    *,
    seed_image: Any,
    seed_text: str,
    reconstruction: Any,
    acceptable_answers: set[Any],
    image_variants: Sequence[Any],
    text_variants: Sequence[str],
    target: Callable[[Any, str], Any],
    normalize: Callable[[Any], Any],
    answer_check: Callable[[Cell], str],
    budget: int,
    image_arm_cap: int,
    text_arm_cap: int,
    joint_budget: int | None = None,
    policy: str = "SF",
    proxies: Proxies | None = None,
    use_proxies: bool = True,
    random_seed: int = 0,
    input_key: Callable[[Any, str], Any] | None = None,
    seed_output: Any = _MISSING,
    reconstruction_output: Any = _MISSING,
    image_search=None,
    text_search=None,
    score_target=None,
    pair_order=None,
) -> DiscoveryResult:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    joint_budget = budget if joint_budget is None else joint_budget
    if joint_budget < 0:
        raise ValueError("joint budget must be nonnegative")
    if image_arm_cap < 0 or text_arm_cap < 0:
        raise ValueError("arm caps must be nonnegative")
    policy = policy.upper()
    if policy == "SAF":
        policy = "SF"
    if policy not in {"RR", "RND", "CRND", "SR", "SF", "JF", "MIX"}:
        raise ValueError(f"unknown pairing policy: {policy}")
    if policy != "MIX" and image_arm_cap + text_arm_cap > budget:
        raise ValueError("single-modality caps exceed the single-modality budget")
    if not acceptable_answers:
        raise ValueError("acceptable_answers must be nonempty")

    result = DiscoveryResult(admitted=False)
    cache: dict[Any, Observation] = {}
    key = input_key or (lambda image, text: (image, text))
    screens = proxies if use_proxies and policy != "RND" and proxies is not None else Proxies()
    raw_seed = (seed_image, seed_text)
    seed = (reconstruction, seed_text)
    rng = random.Random(random_seed)

    def cached(cell: Cell) -> bool:
        return (key(*cell), "predict") in cache

    def call(cell: Cell, purpose: str, request_type="predict") -> Observation:
        cache_key = (key(*cell), request_type)
        if cache_key in cache:
            result.reused += 1
            return cache[cache_key]
        if purpose in {"image_arm", "text_arm", "score"} and result.single_calls >= budget:
            raise RuntimeError("single-modality budget exhausted")
        if purpose == "joint" and result.joint_calls >= joint_budget:
            raise RuntimeError("joint budget exhausted")
        if request_type == "score":
            output = score_target(*cell, result.reconstruction_observation.normalized)
        elif cell == raw_seed and seed_output is not _MISSING:
            output = seed_output
        elif cell == seed and reconstruction_output is not _MISSING:
            output = reconstruction_output
        else:
            output = target(*cell)
        raw_key = key(*cell)
        input_id = raw_key.hex() if isinstance(raw_key, bytes) else repr(raw_key)
        if request_type != "predict":
            input_id = repr((input_id, request_type, result.reconstruction_observation.normalized))
        scores = getattr(target, "scores", None) if request_type == "predict" else None
        if scores is not None:
            normalized_scores = {}
            for answer, value in scores.items():
                answer = normalize(answer)
                normalized_scores[answer] = max(value, normalized_scores.get(answer, float("-inf")))
            scores = normalized_scores
        observation = Observation(cell, output,
                                  normalize(output) if request_type == "predict" else output,
                                  purpose, len(result.log) + 1, input_id, request_type, scores)
        cache[cache_key] = observation
        result.log.append(observation)
        return observation

    call(raw_seed, "raw_seed")
    if answer_check(seed) != "preserved":
        return result
    result.reconstruction_observation = call(seed, "seed")
    if result.reconstruction_observation.normalized not in acceptable_answers:
        return result
    result.admitted = True

    images = []
    seen_image_inputs = {key(*seed)}
    for image in image_variants:
        image_pass = screens.image(image, seed_image)
        pair_pass = screens.pair(image, seed_text)
        arm_key = key(image, seed_text)
        duplicate = arm_key in seen_image_inputs
        passed = image_pass and pair_pass and not duplicate
        result.screening.append({"cell_type": "image_arm", "image": image,
                                 "text": seed_text, "image_pass": image_pass,
                                 "pair_pass": pair_pass, "duplicate": duplicate,
                                 "proxy_pass": image_pass and pair_pass,
                                 "passed": passed})
        if passed:
            images.append(image)
            seen_image_inputs.add(arm_key)
    texts = []
    seen_text_inputs = {key(*seed)}
    for text in text_variants:
        text_pass = screens.text(text, seed_text)
        pair_pass = screens.pair(reconstruction, text)
        arm_key = key(reconstruction, text)
        duplicate = arm_key in seen_text_inputs
        passed = text_pass and pair_pass and not duplicate
        result.screening.append({"cell_type": "text_arm", "image": reconstruction,
                                 "text": text, "text_pass": text_pass,
                                 "pair_pass": pair_pass, "duplicate": duplicate,
                                 "proxy_pass": text_pass and pair_pass,
                                 "passed": passed})
        if passed:
            texts.append(text)
            seen_text_inputs.add(arm_key)

    if policy == "MIX":
        remaining = budget
        total_cap = image_arm_cap + text_arm_cap
        image_limit = (remaining * image_arm_cap // total_cap) if total_cap else remaining // 2
        text_limit = remaining - image_limit
    else:
        image_limit, text_limit = image_arm_cap, text_arm_cap
    def search_pool(search, modality, limit):
        start = result.single_calls
        tested, seen = [], set()

        def evaluate(value, is_test=True):
            cell = (value, seed_text) if modality == "image_arm" else (reconstruction, value)
            identity = key(*cell)
            if is_test:
                relation = screens.image(value, seed_image) if modality == "image_arm" else screens.text(value, seed_text)
                pair_pass = screens.pair(*cell)
                passed = relation and pair_pass
                result.screening.append({"cell_type": modality, "image": cell[0],
                    "text": cell[1], "proxy_pass": passed,
                    "duplicate": identity in seen, "passed": passed})
                if not passed:
                    return float("inf"), None
            if result.single_calls - start >= limit and not cached(cell):
                raise StopIteration
            observation = call(cell, modality if is_test else "score")
            if is_test and identity not in seen and identity != key(*seed):
                tested.append(value)
                seen.add(identity)
            if observation.scores is not None:
                fitness = observation.scores[result.reconstruction_observation.normalized]
            else:
                score_key = (identity, "score")
                if score_key not in cache and result.single_calls - start >= limit:
                    raise StopIteration
                fitness = call(cell, "score", "score").output
            return fitness, observation.normalized

        try:
            search(evaluate, result.reconstruction_observation.normalized)
        except StopIteration:
            pass
        return tested

    images = (search_pool(image_search, "image_arm", image_limit)
              if image_search is not None else images[:image_limit])
    texts = (search_pool(text_search, "text_arm", text_limit)
             if text_search is not None else texts[:text_limit])
    image_arms = [(x, seed_text) for x in images]
    text_arms = [(reconstruction, t) for t in texts]

    if policy != "JF":
        for cell in image_arms + text_arms:
            if result.single_calls >= budget and not cached(cell):
                break
            call(cell, "image_arm" if cell in image_arms else "text_arm")

    if policy != "MIX":
        pairs = list(pair_order(images, texts) if pair_order is not None else _rr_pairs(images, texts))
        if policy in {"RND", "CRND", "SR", "JF"}:
            rng.shuffle(pairs)
        if policy in {"SF", "SR"}:
            stable, other = [], []
            for x, t in pairs:
                arms = ((x, seed_text), (reconstruction, t))
                compatible = all(cached(arm) and
                                 cache[(key(*arm), "predict")].normalized in acceptable_answers
                                 for arm in arms)
                (stable if compatible else other).append((x, t))
            pairs = stable + other

        seen_joint_inputs = set()
        for joint in pairs:
            if result.joint_calls >= joint_budget:
                break
            x, t = joint
            pair_pass = screens.pair(x, t)
            joint_key = key(x, t)
            duplicate = joint_key in seen_joint_inputs
            result.screening.append({"cell_type": "joint", "image": x,
                                     "text": t, "pair_pass": pair_pass,
                                     "duplicate": duplicate,
                                     "proxy_pass": pair_pass,
                                     "passed": pair_pass and not duplicate})
            if not pair_pass or duplicate:
                continue
            seen_joint_inputs.add(joint_key)
            image_arm, text_arm = (x, seed_text), (reconstruction, t)
            needed = sum(not cached(cell) for cell in
                         (image_arm, text_arm))
            if result.single_calls + needed > budget:
                continue
            joint_obs = call(joint, "joint")
            if joint_obs.normalized not in acceptable_answers:
                call(image_arm, "image_arm")
                call(text_arm, "text_arm")
                result.joint_candidates.append(
                    JointCandidate(seed, image_arm, text_arm, joint,
                                   len(result.log)))

    for cell in image_arms:
        if cached(cell) and cache[(key(*cell), "predict")].normalized not in acceptable_answers:
            result.image_candidates.append(cell)
    for cell in text_arms:
        if cached(cell) and cache[(key(*cell), "predict")].normalized not in acceptable_answers:
            result.text_candidates.append(cell)
    return result
