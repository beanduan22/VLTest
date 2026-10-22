# VLTest

This repository implements the relation-constrained VLTest method described in the paper. The discovery entry point is `VLTest/run_vltest.py`; `VLTest/audit.py` handles two-stage human verification. Candidate outputs are provisional until their expected answers and model verdicts have been annotated.

## Method

For each dataset seed and target model, VLTest admits the seed only when the model's answer matches the dataset reference. It records a VQGAN reconstruction control, generates image variants by synchronous replacement of the most frequent VQGAN codebook indices in a fixed bit-reversal rank order, and generates text variants by PEGASUS rewrite, RoBERTa fill-mask expansion, and Sentence-BERT pruning. The ordered pools are deduplicated after target preprocessing. Bidirectional NLI, CLIP image similarity, and task-dependent SigLIP alignment screen variants before target calls. Proxy thresholds must be set from a separate pilot annotation.

One budget covers seed, reconstruction, single-modality arms, and joint inputs. `RR`, `RND`, `SAF`, `JF`, and `MIX` are supported pairing policies. A joint candidate has the seed, image arm, text arm, and joint output in the call log; the joint-first policy reserves enough calls to complete these four observations. A candidate becomes a confirmed failure only after annotation verifies that its answer was preserved and that the model was incorrect. A verified joint failure additionally requires the model to be correct on the seed and both arms.

## Seed manifest

Supply one JSONL row per distinct source image. Image paths can be relative to the manifest. ITM and NLVR2 rows use a Boolean `label`; NLVR2 also requires `right_image`, which remains fixed. VQA rows use `answers`, a list of accepted dataset answers.

```json
{"seed_id":"sample-1","image":"val/example.jpg","text":"Is a dog visible?","answers":["yes","yes","yes"]}
```

## Run discovery

Run from the inner `VLTest/` directory. Replace the symbolic values with the budget, arm caps, rank range, and proxy thresholds fixed for the experiment.

```bash
python run_vltest.py \
  --task vqa --model paligemma --manifest seeds.jsonl --output runs/vqa \
  --vqgan-config configs/coco_scene_images_transformer.yaml \
  --vqgan-checkpoint /path/to/coco_epoch117.ckpt \
  --budget B --image-arm-cap BV --text-arm-cap BT --rank-range R \
  --nli-threshold THETA --image-threshold RHO_IMG \
  --alignment-ratio RHO --negative-margin MU
```

The runner writes `discovery_calls.jsonl`, `seed_summary.jsonl`, `screening.jsonl`, `candidates.jsonl`, and `annotation_reference_blind.jsonl`. The blind sheet contains inputs and dataset seed answers without model outputs. `screening.jsonl` includes accepted and rejected variants for the proxy audit. Each target call is recorded once; a reused observation does not consume another discovery call.

The local target adapters cover ViLT and BLIP for ITM, ViLT and BEiT-3 for NLVR2, and PaliGemma and BEiT-3 for VQA. The paper's additional open-weight and API models require adapters and specified checkpoints before those experiments can run. The runner does not invent values for the paper's unfilled budget and threshold placeholders.

## Dependencies and assets

Install the Python dependencies from `VLTest/environment.yml` and obtain the VQGAN checkpoint, target checkpoints, and auxiliary model weights used in the experiment. The BEiT-3 adapters also require the tokenizer, task checkpoints, and the original VQA `answer2label.txt` under `--assets-dir`. No verified failure counts are supplied by the code without a completed annotation audit.
