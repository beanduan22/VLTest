
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from method import JointCandidate, Observation


@dataclass(frozen=True)
class CellAnnotation:
    expected_answers: frozenset[Any]
    model_correct: bool
    valid: bool = True
    decidable: bool = True


def annotation_outcome(seed_answers: frozenset[Any],
                       annotation: CellAnnotation) -> str:
    if not annotation.valid:
        return "invalid input"
    if not annotation.decidable:
        return "undecidable"
    preserved = annotation.expected_answers == seed_answers
    if preserved:
        return "benign variation" if annotation.model_correct else "confirmed failure"
    return "legitimate change" if annotation.model_correct else "double error"


def confirmed_failure(seed_answers: frozenset[Any], seed: CellAnnotation,
                      variant: CellAnnotation) -> bool:
    return (seed.valid and seed.decidable and seed.model_correct and
            seed.expected_answers == seed_answers and
            annotation_outcome(seed_answers, variant) == "confirmed failure")


def verified_joint_failure(
    candidate: JointCandidate,
    annotations: dict[tuple[Any, str], CellAnnotation],
    seed_answers: frozenset[Any],
) -> bool:
    cells = (candidate.seed, candidate.image_arm,
             candidate.text_arm, candidate.joint)
    if any(cell not in annotations for cell in cells):
        return False
    labels = [annotations[cell] for cell in cells]
    if any(not item.valid or not item.decidable for item in labels):
        return False
    return (bool(seed_answers) and
            all(item.expected_answers == seed_answers for item in labels)
            and all(item.model_correct for item in labels[:3])
            and not labels[3].model_correct)


@dataclass(frozen=True)
class StabilityResult:
    stable: bool
    verification_calls: int
    compatible_votes: dict[tuple[Any, str], int]


def retest_arm_pattern(seed_cell, arm_cell, discovery_log: Iterable[Observation],
                       target, normalize, acceptable_answers: set[Any],
                       *, repeats: int = 3) -> StabilityResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cells = (seed_cell, arm_cell)
    original = {observation.cell: observation for observation in discovery_log}
    if any(cell not in original for cell in cells):
        raise ValueError("arm candidate lacks discovery observations")
    votes = {}
    for cell in cells:
        compatible = int(original[cell].normalized in acceptable_answers)
        for _ in range(repeats):
            compatible += int(normalize(target(*cell)) in acceptable_answers)
        votes[cell] = compatible
    majority = (repeats + 1) // 2 + 1
    stable = votes[seed_cell] >= majority and \
        repeats + 1 - votes[arm_cell] >= majority
    return StabilityResult(stable, repeats * len(cells), votes)


def retest_joint_pattern(candidate: JointCandidate,
                         discovery_log: Iterable[Observation], target, normalize,
                         acceptable_answers: set[Any], *, repeats: int = 3) -> StabilityResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cells = (candidate.seed, candidate.image_arm,
             candidate.text_arm, candidate.joint)
    original = {observation.cell: observation for observation in discovery_log}
    if any(cell not in original for cell in cells):
        raise ValueError("joint candidate lacks discovery observations")
    votes = {}
    for cell in cells:
        compatible = int(original[cell].normalized in acceptable_answers)
        for _ in range(repeats):
            compatible += int(normalize(target(*cell)) in acceptable_answers)
        votes[cell] = compatible
    total_votes = repeats + 1
    majority = total_votes // 2 + 1
    stable = all(votes[cell] >= majority for cell in cells[:3]) and \
        total_votes - votes[cells[3]] >= majority
    return StabilityResult(stable, repeats * len(cells), votes)


@dataclass(frozen=True)
class AuditItem:
    inclusion_probability: float
    human_failure: int
    judge_failure: int = 0


def estimate_human_total(sample: Iterable[AuditItem]) -> float:
    total = 0.0
    for item in sample:
        if not 0 < item.inclusion_probability <= 1:
            raise ValueError("inclusion probability must be in (0, 1]")
        total += item.human_failure / item.inclusion_probability
    return total


def estimate_judge_assisted_total(
    population_judge_positive: int,
    sample: Iterable[AuditItem],
) -> float:
    total = float(population_judge_positive)
    for item in sample:
        if not 0 < item.inclusion_probability <= 1:
            raise ValueError("inclusion probability must be in (0, 1]")
        total += ((item.human_failure - item.judge_failure) /
                  item.inclusion_probability)
    return total


def sample_strata(population: Iterable[Any], stratum_key, sample_sizes: dict,
                  *, random_seed: int) -> list[tuple[Any, float]]:
    import random
    from collections import defaultdict

    groups = defaultdict(list)
    for item in population:
        groups[stratum_key(item)].append(item)
    rng = random.Random(random_seed)
    selected = []
    for key, items in groups.items():
        sample_size = min(sample_sizes.get(key, 0), len(items))
        if sample_size < 0:
            raise ValueError("sample size must be nonnegative")
        selected.extend((item, sample_size / len(items))
                        for item in rng.sample(items, sample_size))
    return selected
