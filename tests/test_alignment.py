import json
import tempfile
import unittest
import types
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import run_vltest

from method import discover
from proxies import make_relation_proxies
from run_vltest import arguments, seed_answers
from targets import load_paper_model, normalize_binary
from verification import VLMOracle, interaction_analysis, retest_joint_pattern
from vision_client import VisionClient
from baselines import TextFoolerSearch, textfooler_recipe


class AlignmentTests(unittest.TestCase):
    def discover(self, **changes):
        values = dict(seed_image="raw", reconstruction="recon", seed_text="q",
                      acceptable_answers={1}, image_variants=["edited"],
                      text_variants=["rewrite"], normalize=int,
                      target=lambda x, t: int((x, t) != ("edited", "rewrite")),
                      answer_check=lambda cell: "preserved", budget=2,
                      image_arm_cap=1, text_arm_cap=1, joint_budget=1)
        values.update(changes)
        return discover(**values)

    def test_four_cells_use_reconstruction_and_test_controls_first(self):
        result = self.discover()
        candidate = result.joint_candidates[0]
        self.assertEqual(candidate.seed, ("recon", "q"))
        self.assertEqual(candidate.text_arm, ("recon", "rewrite"))
        self.assertEqual([item.cell for item in result.log], [
            ("raw", "q"), ("recon", "q"), ("edited", "q"),
            ("recon", "rewrite"), ("edited", "rewrite")])

    def test_admission_uses_reconstruction_correctness(self):
        result = self.discover(target=lambda x, t: int(x != "recon"))
        self.assertFalse(result.admitted)
        self.assertEqual(result.discovery_calls, 0)
        result = self.discover(target=lambda x, t: int(x != "raw"))
        self.assertTrue(result.admitted)

    def test_reconstruction_answer_change_or_abstention_rejects(self):
        for verdict in ["changed", "invalid", "abstain"]:
            with self.subTest(verdict=verdict):
                result = self.discover(answer_check=lambda cell: verdict)
                self.assertFalse(result.admitted)
                self.assertEqual([item.purpose for item in result.log], ["raw_seed"])

    def test_full_single_budget_does_not_consume_joint_budget(self):
        result = self.discover(budget=1000, joint_budget=1000,
                              image_arm_cap=500, text_arm_cap=500,
                              image_variants=[f"x{i}" for i in range(500)],
                              text_variants=[f"t{i}" for i in range(500)])
        purposes = Counter(item.purpose for item in result.log)
        self.assertEqual(purposes, {"raw_seed": 1, "seed": 1,
                                  "image_arm": 500, "text_arm": 500, "joint": 1000})
        self.assertEqual(result.discovery_calls, 2000)

    def test_stable_first_and_random_stable_first_put_compatible_controls_first(self):
        def target(image, text):
            return int(image != "broken" and text != "broken text")
        for policy in ["SF", "SR", "SAF"]:
            result = self.discover(image_variants=["broken", "edited"],
                                  text_variants=["broken text", "rewrite"],
                                  target=target, budget=4, image_arm_cap=2,
                                  text_arm_cap=2, policy=policy)
            joints = [item.cell for item in result.log if item.purpose == "joint"]
            self.assertEqual(joints, [("edited", "rewrite")])

    def test_preprocessed_reconstruction_deduplicates_variants(self):
        result = self.discover(image_variants=["recon alias", "edited"],
                              input_key=lambda x, t: ("recon" if x == "recon alias" else x, t))
        self.assertEqual(result.single_calls, 2)
        self.assertFalse(any(item.cell[0] == "recon alias" for item in result.log))

    def test_pair_filter_uses_reconstruction_baseline(self):
        scores = {("raw", "q"): 0.2, ("recon", "q"): 0.8,
                  ("edited", "q"): 0.7}
        proxies = make_relation_proxies(
            task="itm", seed_image="raw", reconstruction="recon", seed_text="q",
            seed_label=True, entailment_probability=lambda a, b: 1,
            image_similarity=lambda a, b: 1, alignment_score=lambda x, t: scores[x, t],
            nli_threshold=.8, image_threshold=.9, alignment_ratio=.95, negative_margin=.05)
        self.assertFalse(proxies.pair("edited", "q"))

    def test_paper_defaults_and_vqa_full_credit(self):
        args = arguments(["--task", "vqa", "--model", "paligemma", "--manifest", "s.jsonl",
                          "--output", "out", "--vqgan-checkpoint", "gumbel.ckpt"])
        self.assertEqual((args.policy, args.beta, args.rank_range), ("SF", 4000, 8191))
        self.assertEqual((args.budget, args.image_arm_cap, args.text_arm_cap, args.joint_budget),
                         (1000, 500, 500, 1000))
        self.assertEqual(seed_answers({"answers": ["yes"]*4 + ["no"]*3 + ["maybe"]*3},
                                      "vqa", str), {"yes"})
        self.assertEqual(seed_answers({"reference_answers": ["yes"]}, "vqa", str), {"yes"})

    def test_oracle_uses_separate_blind_answer_and_output_requests(self):
        class Client:
            def __init__(self):
                self.requests = []

            def complete(self, prompt, images):
                self.requests.append((prompt, images))
                return json.dumps({"verdict": "incorrect" if '"model_output"' in prompt
                                   else "preserved"})

        client = Client()
        oracle = VLMOracle(client, task="vr", seed=("raw", "q"), answers={1}, right_image="right")
        verdict = oracle.judge(("edited", "rewrite"), 1)
        self.assertTrue(verdict.failure)
        self.assertFalse(verdict.correct)
        self.assertEqual(len(client.requests), 2)
        self.assertNotIn('"model_output"', client.requests[0][0])
        self.assertIn('"model_output": 1', client.requests[1][0])
        self.assertEqual(client.requests[0][1], ["raw", "edited", "right"])

    def test_oracle_abstains_on_unparseable_verdicts(self):
        class Client:
            def complete(self, prompt, images):
                return "not a verdict"

        oracle = VLMOracle(Client(), task="vqa", seed=("raw", "q"), answers={"brown"})
        verdict = oracle.judge(("edited", "rewrite"), "yellow")
        self.assertEqual((verdict.answer, verdict.output), ("abstain", "abstain"))
        self.assertFalse(verdict.failure)

    def test_retest_uses_five_fresh_outputs_and_abstentions_are_not_errors(self):
        result = self.discover()
        seen = []

        def target(image, text):
            seen.append((image, text))
            return int((image, text) != ("edited", "rewrite"))

        outcome = retest_joint_pattern(result.joint_candidates[0], result.log, target, int, {1})
        self.assertTrue(outcome.stable)
        self.assertEqual(Counter(seen), {cell: 5 for cell in [
            ("recon", "q"), ("edited", "q"), ("recon", "rewrite"), ("edited", "rewrite")]})
        outcome = retest_joint_pattern(result.joint_candidates[0], result.log, target, int, {1},
                                       judge=lambda cell, output: "abstain")
        self.assertFalse(outcome.stable)

    def test_additive_margin_and_decoder_disagreement(self):
        cells = [("seed", "q"), ("image", "q"), ("seed", "text"), ("image", "text")]
        margins = dict(zip(cells, [4, 3, 3, -1]))
        result = interaction_analysis(cells, {"right"}, "wrong",
                                       lambda x, t, a: margins[x, t] if a == "right" else 0)
        self.assertTrue(result["interaction_failure"])
        self.assertEqual(result["interaction_contrast"], -3)
        margins[cells[-1]] = 1
        result = interaction_analysis(cells, {"right"}, "wrong",
                                       lambda x, t, a: margins[x, t] if a == "right" else 0)
        self.assertFalse(result["score_classifiable"])
        self.assertIsNone(result["interaction_failure"])

    def test_api_request_models_and_binary_abstention(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}):
            entry = load_paper_model("itm", "gpt-4o-mini", "cpu")
        self.assertEqual(entry["client"].model, "gpt-4o-mini-2024-07-18")
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            image.write_bytes(b"test-image")
            payload = entry["client"].request("caption", [str(image)])
            self.assertEqual(payload["temperature"], 0)
            self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"))
            gemini = VisionClient("gemini-2.5-flash", "https://example.invalid", provider="gemini")
            self.assertEqual(gemini.request("q", [str(image)])["generationConfig"]["temperature"], 0)
        self.assertEqual(normalize_binary("No match."), 0)
        self.assertIsNone(normalize_binary("uncertain"))

    def test_search_filter_rejections_and_classification_scores_share_calls(self):
        seen = []

        def target(image, text):
            seen.append((image, text))
            target.scores = {0: -2., 1: -.1}
            return 1

        def search(evaluate, seed_answer):
            self.assertEqual(evaluate("rejected")[0], float("inf"))
            evaluate("x1")
            evaluate("x2")

        from method import Proxies
        result = self.discover(target=target, budget=4, image_arm_cap=2, text_arm_cap=2,
                              image_variants=[], text_variants=["t1", "t2"],
                              image_search=search, joint_budget=3,
                              proxies=Proxies(image=lambda image, original: image != "rejected"))
        self.assertEqual((result.single_calls, result.joint_calls), (4, 3))
        self.assertNotIn(("rejected", "q"), seen)
        self.assertEqual(len(seen), len(result.log))
        self.assertFalse(any(item.request_type == "score" for item in result.log))

    def test_generative_baseline_scores_are_separate_budgeted_requests(self):
        scored = []

        def score(image, text, answer):
            scored.append((image, text, answer))
            return -.5

        def image_search(evaluate, seed_answer):
            for image in ["x1", "x2"]:
                evaluate(image)

        def text_search(evaluate, seed_answer):
            for text in ["t1", "t2"]:
                evaluate(text)

        result = self.discover(budget=4, image_arm_cap=2, text_arm_cap=2,
                              image_variants=[], text_variants=[],
                              image_search=image_search, text_search=text_search,
                              score_target=score, joint_budget=3)
        self.assertEqual(result.single_calls, 4)
        self.assertEqual(result.joint_calls, 1)
        self.assertEqual(len(scored), 2)
        score_ids = {item.input_id for item in result.log if item.request_type == "score"}
        prediction_ids = {item.input_id for item in result.log if item.request_type == "predict"}
        self.assertFalse(score_ids & prediction_ids)

    def test_textfooler_continues_after_a_failure(self):
        class Text:
            def __init__(self, text):
                self.text = text
                self.num_words = len(text.split())

            def delete_word_at_index(self, index):
                words = self.text.split()
                return Text(" ".join(words[:index] + words[index + 1:]))

        class Recipe:
            def get_transformations(self, text, original_text, indices_to_modify):
                words = text.text.split()
                words[indices_to_modify[0]] += " changed"
                return [Text(" ".join(words))]

        tested = []

        def evaluate(text, is_test=True):
            if is_test:
                tested.append(text)
            return -len(text), 0

        with patch.dict("sys.modules", {"textattack": types.ModuleType("textattack"),
                                       "textattack.shared": types.SimpleNamespace(AttackedText=Text)}):
            TextFoolerSearch("two words", Recipe())(evaluate, 1)
        self.assertEqual(len(tested), 2)

    def test_textfooler_constraint_recipe_does_not_issue_target_calls(self):
        import math
        constraint = type("UniversalSentenceEncoder", (), {})()

        class Recipe:
            @staticmethod
            def build(wrapper):
                self.assertIs(wrapper.model, wrapper)
                with self.assertRaises(RuntimeError):
                    wrapper(["unlogged query"])
                return types.SimpleNamespace(constraints=[constraint])

        modules = {
            "textattack": types.ModuleType("textattack"),
            "textattack.attack_recipes": types.SimpleNamespace(TextFoolerJin2019=Recipe),
            "textattack.models": types.ModuleType("textattack.models"),
            "textattack.models.wrappers": types.SimpleNamespace(ModelWrapper=object),
        }
        textfooler_recipe.cache_clear()
        try:
            with patch.dict("sys.modules", modules):
                textfooler_recipe()
            self.assertAlmostEqual(constraint.threshold, 1 - .7 / math.pi)
        finally:
            textfooler_recipe.cache_clear()

    def test_gumbel_configuration_matches_paper(self):
        import yaml
        path = Path(__file__).parents[1] / "VLTest" / "configs" / "gumbel_f8.yaml"
        model = yaml.safe_load(path.read_text())["model"]
        self.assertEqual(model["target"], "taming.models.vqgan.GumbelVQ")
        self.assertEqual(model["params"]["n_embed"], 8192)
        self.assertEqual(256 // 2**(len(model["params"]["ddconfig"]["ch_mult"]) - 1), 32)

    def test_runner_verifies_candidates_and_records_actual_costs(self):
        import torch

        class Auxiliary:
            mask_token = "<mask>"

            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def load_state_dict(self, *args, **kwargs):
                pass

        class JudgeClient:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def complete(self, prompt, images):
                self.calls += 1
                payload = json.loads(prompt.rsplit("\n", 1)[1])
                if "model_output" not in payload:
                    verdict = "preserved"
                else:
                    verdict = "correct" if payload["model_output"] == 1 else "incorrect"
                return json.dumps({"verdict": verdict})

        fake_transformers = types.ModuleType("transformers")
        for name in ["AutoModelForMaskedLM", "AutoModelForSeq2SeqLM",
                     "AutoModelForSequenceClassification", "AutoTokenizer"]:
            setattr(fake_transformers, name, Auxiliary)
        fake_vqgan = types.ModuleType("taming.models.vqgan")
        fake_vqgan.GumbelVQ = Auxiliary
        fake_modules = {
            "spacy": types.SimpleNamespace(load=lambda *a, **kw: Auxiliary()),
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=Auxiliary),
            "transformers": fake_transformers, "taming.models.vqgan": fake_vqgan,
        }
        for stochastic, broken_control in [(False, False), (True, False), (True, True)]:
            with self.subTest(stochastic=stochastic, broken_control=broken_control):
                seen = Counter()

                def target(image, text):
                    seen[image, text] += 1
                    if image.endswith("raw.png"):
                        return 0
                    if (image, text) == ("edited", "rewrite"):
                        return 0
                    if broken_control and (image, text) == ("edited", "q"):
                        return int(seen[image, text] <= 3)
                    return 1

                def pool(model, raw, output_dir, **kwargs):
                    self.assertTrue(kwargs["admit"]("recon"))
                    return types.SimpleNamespace(reconstruction_path="recon", variants=["edited"])

                entry = {"name": "test", "stochastic": stochastic}
                if stochastic:
                    entry["client"] = types.SimpleNamespace(role="discovery")
                with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                    folder = Path(folder)
                    manifest = folder / "seeds.jsonl"
                    manifest.write_text(json.dumps({"seed_id": "s", "image": "raw.png",
                                                     "text": "q", "label": True}) + "\n")
                    output = folder / "out"
                    argv = ["--task", "itm", "--model", "test", "--manifest", str(manifest),
                            "--output", str(output), "--vqgan-checkpoint", "test.ckpt",
                            "--budget", "2", "--joint-budget", "1", "--disable-proxies"]
                    stack.enter_context(patch.dict("sys.modules", fake_modules))
                    stack.enter_context(patch("sys.argv", ["run_vltest.py"] + argv))
                    stack.enter_context(patch("torch.load", return_value={"state_dict": {}}))
                    stack.enter_context(patch.object(run_vltest, "VisionClient", JudgeClient))
                    stack.enter_context(patch.object(run_vltest, "load_paper_model", return_value=entry))
                    stack.enter_context(patch.object(run_vltest, "make_target", return_value=target))
                    stack.enter_context(patch.object(run_vltest, "make_preprocessed_key",
                        return_value=lambda x, t: repr((x, t)).encode()))
                    stack.enter_context(patch.object(run_vltest, "generate_visual_pool", side_effect=pool))
                    stack.enter_context(patch.object(run_vltest, "pegasus_rewrites", return_value=["rewrite"]))
                    stack.enter_context(patch.object(run_vltest, "generate_text_pool", return_value=["rewrite"]))
                    run_vltest.main()
                    summary_rows = (output / "seed_summary.jsonl").read_text().splitlines()
                    self.assertEqual(len(summary_rows), 1)
                    summary = json.loads(summary_rows[0])
                    outcome = json.loads((output / "oracle_outcomes.jsonl").read_text())
                    self.assertTrue(summary["admitted"])
                    self.assertTrue(outcome["confirmed_failure"])
                    self.assertEqual(outcome["verified_joint_failure"], not broken_control)
                    self.assertEqual((summary["single_calls"], summary["joint_calls"]), (2, 1))
                    self.assertEqual(summary["C_ver"], 20 if stochastic else 0)
                    self.assertEqual(sum(seen.values()), 25 if stochastic else 5)
                    self.assertEqual(len((output / "discovery_calls.jsonl").read_text().splitlines()), 5)
