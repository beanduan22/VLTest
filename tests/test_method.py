import json
import tempfile
import unittest
from pathlib import Path

from audit import (
    estimate_outcomes,
    prepare_verdict_sheet,
    sample_audit,
    summarize_verdicts,
)
from method import Proxies, discover
from verification import (
    AuditItem,
    CellAnnotation,
    estimate_human_total,
    estimate_judge_assisted_total,
    retest_joint_pattern,
    verified_joint_failure,
)
from visual import bit_reversal_ranks, generate_visual_pool


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.answers = {
            ("raw", "question"): 1,
            ("recon", "question"): 1,
            ("edited", "question"): 1,
            ("raw", "rewrite"): 1,
            ("edited", "rewrite"): 0,
        }

    def run_discovery(self, budget=5, policy="RR", proxies=None):
        return discover(
            seed_image="raw", seed_text="question", reconstruction="recon",
            acceptable_answers={1}, image_variants=["edited"],
            text_variants=["rewrite"], target=lambda x, t: self.answers[x, t],
            normalize=int, budget=budget, image_arm_cap=1, text_arm_cap=1,
            policy=policy, proxies=proxies)

    def test_arms_first_four_cell_candidate(self):
        result = self.run_discovery()
        self.assertTrue(result.admitted)
        self.assertEqual(result.discovery_calls, 5)
        self.assertEqual(len(result.joint_candidates), 1)
        self.assertEqual(result.joint_candidates[0].completion_call, 5)
        labels = {
            ("raw", "question"): CellAnnotation(frozenset({1}), True),
            ("edited", "question"): CellAnnotation(frozenset({1}), True),
            ("raw", "rewrite"): CellAnnotation(frozenset({1}), True),
            ("edited", "rewrite"): CellAnnotation(frozenset({1}), False),
        }
        self.assertTrue(verified_joint_failure(result.joint_candidates[0], labels,
                                               frozenset({1})))
        self.assertFalse(verified_joint_failure(result.joint_candidates[0], labels,
                                                frozenset({0})))
        labels[("edited", "rewrite")] = CellAnnotation(frozenset({0}), True)
        self.assertFalse(verified_joint_failure(result.joint_candidates[0], labels,
                                                frozenset({1})))

    def test_precomputed_seed_is_counted_once(self):
        calls = []
        def target(image, text):
            calls.append((image, text))
            return self.answers[image, text]
        result = discover(
            seed_image="raw", seed_text="question", reconstruction="recon",
            acceptable_answers={1}, image_variants=["edited"],
            text_variants=["rewrite"], target=target, normalize=int,
            budget=5, image_arm_cap=1, text_arm_cap=1, seed_output=1)
        self.assertEqual(result.discovery_calls, 5)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.log[0].purpose, "seed")

    def test_identical_reconstruction_reuses_seed_observation(self):
        result = discover(
            seed_image="raw", seed_text="question", reconstruction="raw",
            acceptable_answers={1}, image_variants=[], text_variants=[],
            target=lambda image, text: self.answers[image, text],
            normalize=int, budget=2, image_arm_cap=0, text_arm_cap=0)
        self.assertEqual(result.discovery_calls, 1)
        self.assertIs(result.reconstruction_observation, result.log[0])

    def test_joint_first_reserves_arms(self):
        self.assertEqual(self.run_discovery(4, "JF").discovery_calls, 2)
        result = self.run_discovery(5, "JF")
        self.assertEqual(result.discovery_calls, 5)
        self.assertEqual(len(result.joint_candidates), 1)
        self.assertEqual([record.purpose for record in result.log[2:]],
                         ["joint", "image_arm", "text_arm"])

    def test_joint_retest_needs_majority_in_all_four_cells(self):
        result = self.run_discovery()
        candidate = result.joint_candidates[0]
        outcome = retest_joint_pattern(candidate, result.log,
                                       lambda x, t: self.answers[x, t], int, {1})
        self.assertTrue(outcome.stable)
        self.assertEqual(outcome.verification_calls, 12)
        responses = iter([1, 1, 1] * 3 + [1, 0, 1])
        outcome = retest_joint_pattern(candidate, result.log,
                                       lambda _x, _t: next(responses), int, {1})
        self.assertFalse(outcome.stable)

    def test_pair_proxy_filters_joint(self):
        proxies = Proxies(pair=lambda x, t: (x, t) != ("edited", "rewrite"))
        result = self.run_discovery(proxies=proxies)
        self.assertEqual(result.discovery_calls, 4)
        self.assertFalse(result.joint_candidates)
        self.assertTrue(any(row["cell_type"] == "joint" and not row["passed"]
                            for row in result.screening))

    def test_preprocessed_duplicates_count_once(self):
        result = discover(
            seed_image="raw", seed_text="question", reconstruction="recon",
            acceptable_answers={1}, image_variants=["edited"],
            text_variants=["rewrite", "rewrite alias"],
            target=lambda x, t: self.answers[(x, "rewrite" if t == "rewrite alias" else t)],
            normalize=int, budget=6, image_arm_cap=1, text_arm_cap=2,
            input_key=lambda x, t: (x, "rewrite" if t == "rewrite alias" else t),
        )
        self.assertEqual(result.discovery_calls, 5)
        self.assertEqual(len(result.joint_candidates), 1)
        self.assertTrue(any(row["duplicate"] for row in result.screening))

    def test_audit_estimators(self):
        sample = [AuditItem(0.5, 1, 0), AuditItem(0.5, 0, 1)]
        self.assertEqual(estimate_human_total(sample), 2)
        self.assertEqual(estimate_judge_assisted_total(3, sample), 3)

    def test_reference_is_fixed_before_output_reveal(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            blind = folder / "blind.jsonl"
            calls = folder / "calls.jsonl"
            refs = folder / "refs.jsonl"
            verdict_sheet = folder / "verdict_sheet.jsonl"
            verdict_labels = folder / "verdict_labels.jsonl"
            summary = folder / "summary.jsonl"
            rows = [{"candidate_id": "c1", "seed_id": "s1",
                     "cell_role": role, "input_id": f"id{role}",
                     "image": "raw", "text": "question", "seed_answers": [1]}
                    for role in range(2)]
            blind.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertNotIn("model_output", blind.read_text())
            calls.write_text("\n".join(json.dumps({
                "seed_id": "s1", "input_id": f"id{role}", "output": output})
                for role, output in enumerate([1, 0])) + "\n")
            refs.write_text("\n".join(json.dumps({
                "candidate_id": "c1", "cell_role": role,
                "annotator": f"a{annotator}", "expected_answers": [1],
                "valid": True, "decidable": True})
                for role in range(2) for annotator in range(3)) + "\n")
            prepare_verdict_sheet(blind, calls, refs, verdict_sheet)
            revealed = [json.loads(row) for row in verdict_sheet.read_text().splitlines()]
            self.assertEqual([row["model_output"] for row in revealed], [1, 0])
            verdict_labels.write_text("\n".join(json.dumps({
                "candidate_id": "c1", "cell_role": role,
                "annotator": f"a{annotator}", "model_correct": role == 0})
                for role in range(2) for annotator in range(3)) + "\n")
            summarize_verdicts(verdict_sheet, verdict_labels, summary)
            self.assertEqual(json.loads(summary.read_text())["outcome"],
                             "confirmed failure")

    def test_stratified_audit_estimate(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            population = folder / "population.jsonl"
            sample = folder / "sample.jsonl"
            outcomes = folder / "outcomes.jsonl"
            estimate = folder / "estimate.json"
            rows = [{"candidate_id": f"c{index}", "task": "itm",
                     "model": "m", "policy": "RR", "kind": "joint"}
                    for index in range(4)]
            population.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            sample_audit(population, sample, audit_type="four-cell",
                         sample_size=2, random_seed=1)
            selected = [json.loads(line) for line in sample.read_text().splitlines()]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row["inclusion_probability"] == 0.5
                                for row in selected))
            outcomes.write_text("\n".join(json.dumps({
                "candidate_id": row["candidate_id"],
                "confirmed_failure": True,
                "verified_joint_failure": index == 0})
                for index, row in enumerate(selected)) + "\n")
            estimate_outcomes(population, sample, outcomes, estimate,
                              measure="joint", discovery_calls=100)
            result = json.loads(estimate.read_text())
            self.assertEqual(result["human_estimated_failures"], 2)
            self.assertEqual(result["human_failures_per_1000_calls"], 20)

    def test_bit_reversal_prefix(self):
        self.assertEqual(bit_reversal_ranks(5), [1, 5, 3, 2, 4])
        self.assertEqual(len(set(bit_reversal_ranks(8191))), 8191)

    def test_visual_generator_replaces_top_indices_together(self):
        import torch
        from PIL import Image

        class Quantizer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 3)
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.tensor([
                        [-1., -1., -1.], [-0.5, -0.5, -0.5],
                        [0.5, 0.5, 0.5], [1., 1., 1.]]))

            def get_codebook_entry(self, indices, shape):
                return self.embedding(indices).view(shape).permute(0, 3, 1, 2)

        class VQGAN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quantize = Quantizer()

            def encode(self, _image):
                indices = torch.tensor([0, 0, 1, 1])
                z = self.quantize.get_codebook_entry(indices, (1, 2, 2, 3))
                return z, None, (None, None, indices)

            def decode(self, z):
                return torch.nn.functional.interpolate(z, size=(256, 256), mode="nearest")

        with tempfile.TemporaryDirectory() as folder:
            seed = Path(folder) / "seed.png"
            Image.new("RGB", (256, 256), "white").save(seed)
            pool = generate_visual_pool(VQGAN(), str(seed), str(Path(folder) / "out"),
                                        top_k=2, rank_range=3)
            self.assertEqual(pool.ranks, [1, 3, 2])
            self.assertEqual(len(pool.variants), 3)
            self.assertTrue(Path(pool.reconstruction_path).exists())


if __name__ == "__main__":
    unittest.main()
