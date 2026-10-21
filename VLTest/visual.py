
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def bit_reversal_ranks(rank_range: int) -> list[int]:
    if rank_range < 1:
        raise ValueError("rank_range must be positive")
    width = (rank_range - 1).bit_length()
    result = []
    for number in range(1 << width):
        reversed_bits = int(f"{number:0{width}b}"[::-1], 2) if width else 0
        rank = reversed_bits + 1
        if rank <= rank_range:
            result.append(rank)
    return result


@dataclass
class VisualPool:
    raw_seed_path: str
    reconstruction_path: str
    variants: list[str]
    ranks: list[int]


def generate_visual_pool(
    vqgan,
    raw_seed_path: str,
    output_dir: str,
    *,
    top_k: int,
    rank_range: int,
    max_variants: int | None = None,
    model_input_key: Callable[[str], bytes] | None = None,
    admit: Callable[[str], bool] | None = None,
) -> VisualPool:
    import numpy as np
    import torch
    from PIL import Image

    if top_k < 1:
        raise ValueError("top_k must be positive")
    device = next(vqgan.parameters()).device
    image = Image.open(raw_seed_path).convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)),
                           Image.Resampling.LANCZOS)
    left = (resized.width - 256) // 2
    top = (resized.height - 256) // 2
    cropped = resized.crop((left, top, left + 256, top + 256))
    tensor = torch.from_numpy(np.asarray(cropped).copy()).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def save(decoded, path: Path) -> None:
        rgb = ((decoded[0].detach().clamp(-1, 1) + 1) * 127.5)
        rgb = rgb.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        Image.fromarray(rgb, mode="RGB").save(path)

    with torch.no_grad():
        quantized, _, (_, _, indices) = vqgan.encode(tensor)
        recon_path = output / "reconstruction.png"
        save(vqgan.decode(quantized), recon_path)
        if admit is not None and not admit(str(recon_path)):
            return VisualPool(str(raw_seed_path), str(recon_path), [], [])

        original = indices.reshape(-1)
        counts = Counter(original.cpu().tolist())
        selected = [index for index, _ in
                    sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
        embedding = getattr(vqgan.quantize, "embed", None)
        if embedding is None:
            embedding = vqgan.quantize.embedding
        codebook = embedding.weight.detach().cpu().numpy()
        size, dimension = codebook.shape
        if rank_range > size - 1:
            raise ValueError("rank_range exceeds the codebook")
        neighbours = {}
        all_ids = np.arange(size)
        for old_id in selected:
            distances = np.linalg.norm(codebook - codebook[old_id], axis=1)
            order = np.lexsort((all_ids, distances))
            neighbours[old_id] = order[order != old_id]

        h, w = quantized.shape[-2:]
        ranks = bit_reversal_ranks(rank_range)
        if max_variants is not None:
            ranks = ranks[:max_variants]
        seen_grids: set[bytes] = set()
        seen_inputs: set[bytes] = set()
        variants, retained_ranks = [], []
        for rank in ranks:
            edited = original.clone()
            for old_id in selected:
                edited[original == old_id] = int(neighbours[old_id][rank - 1])
            grid_key = hashlib.blake2b(edited.cpu().numpy().tobytes(),
                                        digest_size=16).digest()
            if grid_key in seen_grids:
                continue
            seen_grids.add(grid_key)
            z = vqgan.quantize.get_codebook_entry(
                edited, (1, h, w, dimension))
            path = output / f"rank_{rank:05d}.png"
            save(vqgan.decode(z), path)
            model_bytes = model_input_key(str(path)) if model_input_key else path.read_bytes()
            input_key = hashlib.blake2b(model_bytes, digest_size=16).digest()
            if input_key in seen_inputs:
                path.unlink()
                continue
            seen_inputs.add(input_key)
            variants.append(str(path))
            retained_ranks.append(rank)
    return VisualPool(str(raw_seed_path), str(recon_path), variants, retained_ranks)
