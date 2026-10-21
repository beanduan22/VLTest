from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path


class SINVADSearch:
    def __init__(self, vqgan, image_path, output_dir, *, random_seed):
        self.vqgan = vqgan
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed

    def __call__(self, evaluate, seed_answer):
        import numpy as np
        import torch
        from PIL import Image

        device = next(self.vqgan.parameters()).device
        generator = torch.Generator(device=device).manual_seed(self.random_seed)
        with Image.open(self.image_path) as opened:
            image = opened.convert("RGB")
        scale = 256 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                              Image.Resampling.LANCZOS)
        left, top = (image.width - 256) // 2, (image.height - 256) // 2
        pixels = torch.from_numpy(np.asarray(image.crop((left, top, left + 256, top + 256))).copy())
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 127.5 - 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            original = self.vqgan.quant_conv(self.vqgan.encoder(pixels))
            shape = original.shape
            original = original.flatten()
            population = [original + .7 * torch.randn(original.shape, generator=generator, device=device)
                          for _ in range(50)]
            fitness_cache = {}
            mutation, previous = .1, float("inf")
            for generation in range(500):
                fitness = []
                for index, member in enumerate(population):
                    identity = member.detach().cpu().numpy().tobytes()
                    if identity not in fitness_cache:
                        quantized = self.vqgan.quantize(member.reshape(shape))[0]
                        decoded = self.vqgan.decode(quantized)[0].clamp(-1, 1)
                        rgb = ((decoded + 1) * 127.5).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                        path = self.output_dir / f"{generation:03d}_{index:02d}.png"
                        Image.fromarray(rgb).save(path)
                        fitness_cache[identity] = evaluate(str(path))[0]
                    fitness.append(fitness_cache[identity])
                best_indices = sorted(range(50), key=lambda i: fitness[i])[:20]
                best = fitness[best_indices[0]]
                mutation = mutation * .7 if best == previous else .1
                if mutation < .001:
                    break
                parents = [population[index] for index in best_indices]
                children = []
                for _ in range(30):
                    mother, father = torch.randperm(20, generator=generator, device=device)[:2].tolist()
                    split = int(torch.randint(original.numel(), (1,), generator=generator, device=device))
                    child = torch.cat((parents[mother][:split], parents[father][split:])).clone()
                    noise = torch.randn(child.shape, generator=generator, device=device)
                    child += mutation * noise * child.ne(original)
                    mask = torch.rand(child.shape, generator=generator, device=device) < .5
                    children.append(torch.where(mask, original, child))
                population = parents + children
                previous = best


class TextFoolerSearch:
    def __init__(self, text, recipe=None):
        self.text = text
        self.recipe = recipe

    def __call__(self, evaluate, seed_answer):
        from textattack.shared import AttackedText

        recipe = self.recipe or textfooler_recipe()
        original = AttackedText(self.text)
        current = original
        seed_score, _ = evaluate(self.text, False)
        importance = []
        for index in range(original.num_words):
            score, _ = evaluate(original.delete_word_at_index(index).text, False)
            importance.append((seed_score - score, index))
        current_score = seed_score
        for _, index in sorted(importance, key=lambda item: (-item[0], item[1])):
            transformations = recipe.get_transformations(
                current, original_text=original, indices_to_modify=[index])
            best, best_score = current, current_score
            for transformed in transformations:
                score, _ = evaluate(transformed.text)
                if score < best_score:
                    best, best_score = transformed, score
            current, current_score = best, best_score


@lru_cache(maxsize=1)
def textfooler_recipe():
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.models.wrappers import ModelWrapper

    class Wrapper(ModelWrapper):
        def __init__(self):
            self.model = self

        def __call__(self, texts):
            raise RuntimeError("target calls must go through the discovery call log")

    recipe = TextFoolerJin2019.build(Wrapper())
    for constraint in recipe.constraints:
        if type(constraint).__name__ == "UniversalSentenceEncoder":
            constraint.threshold = 1 - .7 / math.pi
    return recipe
