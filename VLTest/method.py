
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
        return len(self.log)


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
    budget: int,
    image_arm_cap: int,
    text_arm_cap: int,
    policy: str = "RR",
    proxies: Proxies | None = None,
    use_proxies: bool = True,
    random_seed: int = 0,
    input_key: Callable[[Any, str], Any] | None = None,
    seed_output: Any = _MISSING,
) -> DiscoveryResult:
    if budget < 1:
        raise ValueError("budget must include at least the seed call")
    if image_arm_cap < 0 or text_arm_cap < 0:
        raise ValueError("arm caps must be nonnegative")
    policy = policy.upper()
    if policy not in {"RR", "RND", "SAF", "JF", "MIX"}:
        raise ValueError(f"unknown pairing policy: {policy}")
    if not acceptable_answers:
        raise ValueError("acceptable_answers must be nonempty")

    result = DiscoveryResult(admitted=False)
    cache: dict[Any, Observation] = {}
    key = input_key or (lambda image, text: (image, text))
    screens = proxies if use_proxies and proxies is not None else Proxies()
    seed = (seed_image, seed_text)
    rng = random.Random(random_seed)

    def cached(cell: Cell) -> bool:
        return key(*cell) in cache

    def call(cell: Cell, purpose: str) -> Observation:
        cache_key = key(*cell)
        if cache_key in cache:
            result.reused += 1
            return cache[cache_key]
        if result.discovery_calls >= budget:
            raise RuntimeError("discovery budget exhausted")
        output = seed_output if cell == seed and seed_output is not _MISSING \
            else target(*cell)
        input_id = cache_key.hex() if isinstance(cache_key, bytes) else repr(cache_key)
        observation = Observation(cell, output, normalize(output), purpose,
                                  result.discovery_calls + 1, input_id)
        cache[cache_key] = observation
        result.log.append(observation)
        return observation

    if call(seed, "seed").normalized not in acceptable_answers:
        return result
    result.admitted = True
    if result.discovery_calls < budget:
        result.reconstruction_observation = call(
            (reconstruction, seed_text), "reconstruction")

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
        pair_pass = screens.pair(seed_image, text)
        arm_key = key(seed_image, text)
        duplicate = arm_key in seen_text_inputs
        passed = text_pass and pair_pass and not duplicate
        result.screening.append({"cell_type": "text_arm", "image": seed_image,
                                 "text": text, "text_pass": text_pass,
                                 "pair_pass": pair_pass, "duplicate": duplicate,
                                 "proxy_pass": text_pass and pair_pass,
                                 "passed": passed})
        if passed:
            texts.append(text)
            seen_text_inputs.add(arm_key)

    if policy == "MIX":
        remaining = max(0, budget - result.discovery_calls)
        total_cap = image_arm_cap + text_arm_cap
        image_limit = (remaining * image_arm_cap // total_cap) if total_cap else remaining // 2
        text_limit = remaining - image_limit
    else:
        image_limit, text_limit = image_arm_cap, text_arm_cap
    images = images[:image_limit]
    texts = texts[:text_limit]
    image_arms = [(x, seed_text) for x in images]
    text_arms = [(seed_image, t) for t in texts]

    if policy != "JF":
        for cell in _interleave(image_arms, text_arms):
            if result.discovery_calls >= budget and not cached(cell):
                break
            call(cell, "image_arm" if cell[0] != seed_image else "text_arm")

    if policy != "MIX":
        pairs = list(_rr_pairs(images, texts))
        if policy in {"RND", "JF"}:
            rng.shuffle(pairs)
        elif policy == "SAF":
            stable, other = [], []
            for x, t in pairs:
                arms = ((x, seed_text), (seed_image, t))
                compatible = all(cached(arm) and
                                 cache[key(*arm)].normalized in acceptable_answers
                                 for arm in arms)
                (stable if compatible else other).append((x, t))
            pairs = stable + other

        seen_joint_inputs = set()
        for joint in pairs:
            if result.discovery_calls >= budget:
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
            image_arm, text_arm = (x, seed_text), (seed_image, t)
            needed = sum(not cached(cell) for cell in
                         (joint, image_arm, text_arm))
            if result.discovery_calls + needed > budget:
                continue
            joint_obs = call(joint, "joint")
            if joint_obs.normalized not in acceptable_answers:
                call(image_arm, "image_arm")
                call(text_arm, "text_arm")
                result.joint_candidates.append(
                    JointCandidate(seed, image_arm, text_arm, joint,
                                   result.discovery_calls))

    for cell in image_arms:
        if cached(cell) and cache[key(*cell)].normalized not in acceptable_answers:
            result.image_candidates.append(cell)
    for cell in text_arms:
        if cached(cell) and cache[key(*cell)].normalized not in acceptable_answers:
            result.text_candidates.append(cell)
    return result
