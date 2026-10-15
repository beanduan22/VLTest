
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from verification import (
    AuditItem,
    CellAnnotation,
    annotation_outcome,
    estimate_human_total,
    estimate_judge_assisted_total,
    sample_strata,
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_audit(input_path, output_path, *, audit_type, sample_size,
                 random_seed):
    rows = read_jsonl(input_path)
    if audit_type == "four-cell":
        rows = [row for row in rows if row["kind"] == "joint"]
    elif audit_type == "proxy":
        rows = [row for row in rows if not row["duplicate"]]
    elif audit_type != "candidate":
        raise ValueError(f"unknown audit type: {audit_type}")
    if audit_type == "proxy":
        def stratum(row):
            return (row["task"], row["model"], row["cell_type"],
                    row.get("policy"), row["proxy_pass"])
    else:
        def stratum(row):
            return (row["task"], row["model"], row["policy"], row["kind"],
                    row.get("proxies_enabled"))
    keys = {stratum(row) for row in rows}
    selected = sample_strata(rows, stratum,
                             {key: sample_size for key in keys},
                             random_seed=random_seed)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps({**row, "inclusion_probability": probability},
                                    ensure_ascii=False) + "\n" for row, probability in selected)


def majority_reference(records):
    regular = [row for row in records if row["annotator"] != "adjudication"]
    if len({row["annotator"] for row in regular}) < 3:
        raise ValueError("three independent reference annotators are required")
    votes = Counter((bool(row["valid"]), bool(row["decidable"]),
                     tuple(sorted(row["expected_answers"]))) for row in regular)
    winner, count = votes.most_common(1)[0]
    if count < 2:
        adjudicated = [row for row in records if row["annotator"] == "adjudication"]
        if len(adjudicated) != 1:
            raise ValueError("three-way reference split needs one adjudication")
        row = adjudicated[0]
        winner = (bool(row["valid"]), bool(row["decidable"]),
                  tuple(sorted(row["expected_answers"])))
    return winner


def prepare_verdict_sheet(blind_path, calls_path, reference_labels_path,
                          output_path, sample_path=None):
    blind = read_jsonl(blind_path)
    if sample_path is not None:
        selected = {row["candidate_id"] for row in read_jsonl(sample_path)}
        blind = [row for row in blind if row["candidate_id"] in selected]
    calls = read_jsonl(calls_path)
    labels = read_jsonl(reference_labels_path)
    grouped = defaultdict(list)
    for row in labels:
        grouped[(row["candidate_id"], row["cell_role"])].append(row)
    outputs = {(row["seed_id"], row["input_id"]): row["output"]
               for row in calls}
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for row in blind:
            identity = (row["candidate_id"], row["cell_role"])
            valid, decidable, answers = majority_reference(grouped[identity])
            output_key = (row["seed_id"], row["input_id"])
            if output_key not in outputs:
                raise ValueError(f"missing discovery output for {identity}")
            handle.write(json.dumps({
                **row, "expected_answers": list(answers),
                "valid": valid, "decidable": decidable,
                "model_output": outputs[output_key],
            }, ensure_ascii=False) + "\n")


def summarize_verdicts(verdict_sheet_path, verdict_labels_path, output_path):
    sheet = read_jsonl(verdict_sheet_path)
    labels = read_jsonl(verdict_labels_path)
    grouped = defaultdict(list)
    for row in labels:
        grouped[(row["candidate_id"], row["cell_role"])].append(row)
    by_candidate = defaultdict(dict)
    seed_answer_by_candidate = {}
    for row in sheet:
        identity = (row["candidate_id"], row["cell_role"])
        votes = [bool(item["model_correct"]) for item in grouped[identity]
                 if item["annotator"] != "adjudication"]
        if len(votes) < 3:
            raise ValueError(f"three correctness verdicts required for {identity}")
        correct = sum(votes) >= 2
        by_candidate[row["candidate_id"]][row["cell_role"]] = CellAnnotation(
            frozenset(row["expected_answers"]), correct,
            bool(row["valid"]), bool(row["decidable"]))
        if row["cell_role"] == 0:
            if row["candidate_id"] in seed_answer_by_candidate:
                raise ValueError(f"duplicate seed record for {row['candidate_id']}")
            seed_answer_by_candidate[row["candidate_id"]] = frozenset(
                row["seed_answers"])

    with Path(output_path).open("w", encoding="utf-8") as handle:
        for candidate_id, cells in by_candidate.items():
            if 0 not in cells:
                raise ValueError(f"missing seed cell for {candidate_id}")
            seed_answers = seed_answer_by_candidate[candidate_id]
            variant_role = 3 if len(cells) == 4 else 1
            variant = cells[variant_role]
            confirmed = (cells[0].valid and cells[0].decidable and
                         cells[0].expected_answers == seed_answers and
                         cells[0].model_correct and variant.valid and
                         variant.decidable and
                         variant.expected_answers == seed_answers and
                         not variant.model_correct)
            if len(cells) == 4:
                labels = [cells[index] for index in range(4)]
                if any(not item.valid or not item.decidable for item in labels):
                    outcome = "invalid or undecidable"
                elif labels[0].expected_answers != seed_answers:
                    outcome = "seed reference changed"
                elif not labels[0].model_correct:
                    outcome = "seed incorrect"
                elif any(item.expected_answers != labels[0].expected_answers
                         for item in labels[1:3]):
                    outcome = "arm answer changed"
                elif not all(item.model_correct for item in labels[1:3]):
                    outcome = "arm incorrect"
                elif labels[3].expected_answers != labels[0].expected_answers:
                    outcome = "joint answer changed"
                elif labels[3].model_correct:
                    outcome = "joint model correct"
                else:
                    outcome = "verified joint failure"
            elif len(cells) == 2:
                outcome = (annotation_outcome(cells[0].expected_answers, cells[1])
                           if cells[0].valid and cells[0].decidable
                           and cells[0].model_correct
                           and cells[0].expected_answers == seed_answers
                           else "seed incorrect")
            else:
                raise ValueError(f"incomplete candidate: {candidate_id}")
            handle.write(json.dumps({"candidate_id": candidate_id,
                                     "outcome": outcome,
                                     "confirmed_failure": confirmed,
                                     "verified_joint_failure":
                                     outcome == "verified joint failure"},
                                    ensure_ascii=False) + "\n")


def estimate_outcomes(population_path, sample_path, outcomes_path, output_path,
                      *, measure, judge_path=None, discovery_calls=None):
    population = read_jsonl(population_path)
    sample = read_jsonl(sample_path)
    if measure == "joint":
        population = [row for row in population if row["kind"] == "joint"]
        sample = [row for row in sample if row["kind"] == "joint"]
    outcomes = {row["candidate_id"]: row
                for row in read_jsonl(outcomes_path)}
    population_ids = [row["candidate_id"] for row in population]
    if len(population_ids) != len(set(population_ids)):
        raise ValueError("population candidate IDs must be unique")
    population_set = set(population_ids)
    judge = None
    if judge_path is not None:
        judge = {row["candidate_id"]: int(row["failure"])
                 for row in read_jsonl(judge_path)}
        if not population_set.issubset(judge):
            raise ValueError("judge labels must cover the full population")
        judge = {candidate_id: judge[candidate_id]
                 for candidate_id in population_ids}
    items = []
    for row in sample:
        candidate_id = row["candidate_id"]
        if candidate_id not in population_set or candidate_id not in outcomes:
            raise ValueError(f"missing sampled candidate: {candidate_id}")
        positive = ("verified_joint_failure" if measure == "joint"
                    else "confirmed_failure")
        items.append(AuditItem(
            float(row["inclusion_probability"]),
            int(outcomes[candidate_id][positive]),
            judge[candidate_id] if judge is not None else 0))
    result = {"measure": measure, "population_candidates": len(population),
              "sampled_candidates": len(sample),
              "human_estimated_failures": estimate_human_total(items)}
    if judge is not None:
        result["judge_assisted_estimated_failures"] = (
            estimate_judge_assisted_total(sum(judge.values()), items))
    if discovery_calls is not None:
        if discovery_calls <= 0:
            raise ValueError("discovery calls must be positive")
        result["human_failures_per_1000_calls"] = (
            1000 * result["human_estimated_failures"] / discovery_calls)
        if judge is not None:
            result["judge_assisted_failures_per_1000_calls"] = (
                1000 * result["judge_assisted_estimated_failures"] /
                discovery_calls)
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    sample = subcommands.add_parser("sample")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--audit-type", choices=["candidate", "four-cell", "proxy"],
                        required=True)
    sample.add_argument("--sample-size", type=int, required=True)
    sample.add_argument("--seed", type=int, default=123456)
    prepare = subcommands.add_parser("reveal")
    prepare.add_argument("--blind", required=True)
    prepare.add_argument("--calls", required=True)
    prepare.add_argument("--reference-labels", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--sample")
    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--verdict-sheet", required=True)
    summarize.add_argument("--verdict-labels", required=True)
    summarize.add_argument("--output", required=True)
    estimate = subcommands.add_parser("estimate")
    estimate.add_argument("--population", required=True)
    estimate.add_argument("--sample", required=True)
    estimate.add_argument("--outcomes", required=True)
    estimate.add_argument("--output", required=True)
    estimate.add_argument("--measure", choices=["confirmed", "joint"],
                          required=True)
    estimate.add_argument("--judge")
    estimate.add_argument("--discovery-calls", type=int)
    args = parser.parse_args()
    if args.command == "sample":
        sample_audit(args.input, args.output, audit_type=args.audit_type,
                     sample_size=args.sample_size, random_seed=args.seed)
    elif args.command == "reveal":
        prepare_verdict_sheet(args.blind, args.calls, args.reference_labels,
                              args.output, args.sample)
    elif args.command == "summarize":
        summarize_verdicts(args.verdict_sheet, args.verdict_labels, args.output)
    else:
        estimate_outcomes(args.population, args.sample, args.outcomes,
                          args.output, measure=args.measure, judge_path=args.judge,
                          discovery_calls=args.discovery_calls)


if __name__ == "__main__":
    main()
