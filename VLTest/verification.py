
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
import json

from method import JointCandidate, Observation

_NO_OUTPUT = object()


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
    incorrect_votes: dict[tuple[Any, str], int] | None = None


def retest_arm_pattern(seed_cell, arm_cell, discovery_log: Iterable[Observation],
                       target, normalize, acceptable_answers: set[Any],
                       *, repeats: int = 5, judge=None, input_key=None) -> StabilityResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cells = (seed_cell, arm_cell)
    key = input_key or (lambda image, text: (image, text))
    original = {key(*observation.cell): observation for observation in discovery_log}
    if any(key(*cell) not in original for cell in cells):
        raise ValueError("arm candidate lacks discovery observations")
    votes, incorrect_votes = {}, {}
    for cell in cells:
        compatible = incorrect = 0
        for _ in range(repeats):
            output = target(*cell)
            verdict = (judge(cell, output) if judge is not None else
                       ("correct" if normalize(output) in acceptable_answers else "incorrect"))
            compatible += int(verdict == "correct")
            incorrect += int(verdict == "incorrect")
        votes[cell] = compatible
        incorrect_votes[cell] = incorrect
    majority = repeats // 2 + 1
    stable = votes[seed_cell] >= majority and \
        incorrect_votes[arm_cell] >= majority
    return StabilityResult(stable, repeats * len(cells), votes, incorrect_votes)


def retest_joint_pattern(candidate: JointCandidate,
                         discovery_log: Iterable[Observation], target, normalize,
                         acceptable_answers: set[Any], *, repeats: int = 5,
                         judge=None, input_key=None) -> StabilityResult:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cells = (candidate.seed, candidate.image_arm,
             candidate.text_arm, candidate.joint)
    key = input_key or (lambda image, text: (image, text))
    original = {key(*observation.cell): observation for observation in discovery_log}
    if any(key(*cell) not in original for cell in cells):
        raise ValueError("joint candidate lacks discovery observations")
    votes, incorrect_votes = {}, {}
    for cell in cells:
        compatible = incorrect = 0
        for _ in range(repeats):
            output = target(*cell)
            verdict = (judge(cell, output) if judge is not None else
                       ("correct" if normalize(output) in acceptable_answers else "incorrect"))
            compatible += int(verdict == "correct")
            incorrect += int(verdict == "incorrect")
        votes[cell] = compatible
        incorrect_votes[cell] = incorrect
    total_votes = repeats
    majority = total_votes // 2 + 1
    stable = all(votes[cell] >= majority for cell in cells[:3]) and \
        incorrect_votes[cells[3]] >= majority
    return StabilityResult(stable, repeats * len(cells), votes, incorrect_votes)


@dataclass(frozen=True)
class OracleVerdict:
    answer: str
    output: str

    @property
    def correct(self):
        return self.answer == "preserved" and self.output == "correct"

    @property
    def failure(self):
        return self.answer == "preserved" and self.output == "incorrect"


class VLMOracle:
    def __init__(self, client, *, task, seed, answers, right_image=None):
        self.client = client
        self.task = task
        self.seed = seed
        self.answers = answers
        self.right_image = right_image
        self.answers_cache = {}
        self.outputs_cache = {}

    def _request(self, cell, instruction, output=_NO_OUTPUT):
        reference = {"task": self.task, "seed_text": self.seed[1],
                     "reference_answers": sorted(self.answers), "generated_text": cell[1]}
        if output is not _NO_OUTPUT:
            reference["model_output"] = output
        images = [self.seed[0], cell[0]]
        image_order = "Images are the raw seed image, then the generated image."
        if self.right_image is not None:
            images.append(self.right_image)
            image_order += " The last image is the fixed second image for both VR inputs."
        prompt = (instruction + " " + image_order + "\n" + json.dumps(reference))
        response = self.client.complete(prompt, images)
        try:
            content = response.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            verdict = json.loads(content)["verdict"]
        except (ValueError, KeyError, TypeError):
            verdict = "abstain"
        return verdict if isinstance(verdict, str) else "abstain"

    def answer_check(self, cell):
        if cell not in self.answers_cache:
            verdict = self._request(cell,
                'Determine whether the generated input is meaningful for the task and '
                'has the same expected answer as the seed, using the seed and reference '
                'answers as the reference. Check image details, counts, relations and '
                'the meaning of the text. Do not assume that a mutation preserves the '
                'answer. No model output is supplied. Return only a JSON object with '
                '"verdict": "preserved", "changed", "invalid", or "abstain".')
            self.answers_cache[cell] = (verdict if verdict in
                {"preserved", "changed", "invalid", "abstain"} else "abstain")
        return self.answers_cache[cell]

    def output_check(self, cell, output):
        identity = (cell, json.dumps(output, sort_keys=True))
        if identity not in self.outputs_cache:
            verdict = self._request(cell,
                'Judge whether model_output is a correct answer to the generated '
                'input. Use the seed and reference answers as a reference, but '
                'judge the actual generated input. Accept correct synonyms and '
                'equivalent wording outside the reference answers. Return only a '
                'JSON object with "verdict": "correct", "incorrect", or "abstain".',
                output)
            if self.task in {"itm", "vr", "nlvr"}:
                from targets import normalize_binary
                if normalize_binary(output) is None:
                    verdict = "abstain"
            self.outputs_cache[identity] = (verdict if verdict in
                {"correct", "incorrect", "abstain"} else "abstain")
        return self.outputs_cache[identity]

    def judge(self, cell, output):
        answer = self.answer_check(cell)
        return OracleVerdict(answer, self.output_check(cell, output))


def interaction_analysis(cells, reference_answers, joint_answer, score):
    margins = []
    for image, text in cells:
        reference = max(score(image, text, answer) for answer in reference_answers)
        margins.append(reference - score(image, text, joint_answer))
    seed, image, text, joint = margins
    additive = image + text - seed
    return {"margins": margins, "additive_margin": additive,
            "interaction_contrast": joint - image - text + seed,
            "score_classifiable": joint <= 0,
            "interaction_failure": additive > 0 if joint <= 0 else None}


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
