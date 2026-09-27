from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    s = float(v.std())
    return (v - float(v.mean())) / (s + 1e-6)


def brain_condition(fly_dir: Path, prompt: str, duration_ms: float) -> dict:
    fly_dir = fly_dir.resolve()
    sys.path.insert(0, str(fly_dir))
    os.chdir(fly_dir)
    from flybrain.model import FlyBrain

    t0 = time.time()
    brain = FlyBrain()
    build_s = time.time() - t0
    trial = brain.run_trial("sugar", duration_ms=duration_ms)

    spikes_i = np.asarray(trial.spikes_i, dtype=np.int32)
    spikes_t = np.asarray(trial.spikes_t_ms, dtype=np.float32)
    n_neurons = int(brain.n_neurons)
    n_synapses = int(brain.n_synapses)

    # Rich population statistics from the real whole-brain spike response.
    neuron_hist, _ = np.histogram(spikes_i, bins=64, range=(0, max(1, n_neurons)))
    time_hist, _ = np.histogram(spikes_t, bins=64, range=(0.0, max(duration_ms, 1.0)))
    neuron_hist = _norm(np.log1p(neuron_hist))
    time_hist = _norm(np.log1p(time_hist))

    # Active-neuron identity histogram retains information that simple spike counts lose.
    if spikes_i.size:
        active = np.unique(spikes_i)
        active_ids = np.asarray(brain.i2flyid[active], dtype=np.int64)
        id_mod = (active_ids % 65536).astype(np.float32)
        id_hist, _ = np.histogram(id_mod, bins=64, range=(0, 65536))
        id_hist = _norm(np.log1p(id_hist))
    else:
        active = np.empty(0, dtype=np.int32)
        active_ids = np.empty(0, dtype=np.int64)
        id_hist = np.zeros(64, dtype=np.float32)

    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(np.asarray([n_neurons, n_synapses], dtype=np.int64).tobytes())
    h.update(spikes_i.tobytes())
    h.update(np.rint(spikes_t * 1000).astype(np.int32).tobytes())
    h.update(active_ids.tobytes())
    digest = h.digest()
    seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)

    # Brain activity controls prompt-level style in addition to direct latent conditioning.
    light = [
        "soft north-window light", "cinematic rim light", "golden-hour light",
        "cool overcast daylight", "large studio softbox", "dramatic side light",
    ][digest[8] % 6]
    lens = [
        "85mm portrait photography", "50mm prime photography", "shallow depth of field",
        "crisp documentary photography", "macro fur detail",
    ][digest[9] % 5]
    palette = [
        "natural color science", "warm neutral palette", "cool neutral palette",
        "rich realistic colors",
    ][digest[10] % 4]

    result = {
        "seed": seed,
        "digest": digest.hex(),
        "style": f"{light}, {lens}, {palette}",
        "n_neurons": n_neurons,
        "n_synapses": n_synapses,
        "build_seconds": build_s,
        "trial_sim_ms": float(trial.sim_ms),
        "trial_wall_ms": float(trial.wall_ms),
        "active_neurons": int(trial.active_neurons),
        "spike_count": int(spikes_i.size),
        "mn9_spikes": int(trial.mn9_spikes),
        "eat": bool(trial.extra.get("eat", False)),
        "neuron_hist": neuron_hist.tolist(),
        "time_hist": time_hist.tolist(),
        "id_hist": id_hist.tolist(),
    }

    # Free the full connectome before loading diffusion weights.
    del brain, trial, spikes_i, spikes_t, active, active_ids
    gc.collect()
    return result


def make_brain_latent(condition: dict, candidate: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Map measured spike population structure directly into SD's initial latent."""
    nh = np.asarray(condition["neuron_hist"], dtype=np.float32)
    th = np.asarray(condition["time_hist"], dtype=np.float32)
    ih = np.asarray(condition["id_hist"], dtype=np.float32)

    # Four channels use different circular shifts so the brain signal is not duplicated.
    maps = []
    for c in range(4):
        a = np.roll(nh, (candidate * 7 + c * 11) % 64)
        b = np.roll(th, (candidate * 13 + c * 5) % 64)
        d = np.roll(ih, (candidate * 3 + c * 17) % 64)
        m = np.outer(a, b) + 0.55 * np.outer(d, np.roll(b, c * 9))
        m = _norm(m)
        maps.append(m)
    return np.stack(maps, axis=0).astype(np.float32)


def sharpness_score(image: Image.Image) -> float:
    g = np.asarray(image.convert("L"), dtype=np.float32)
    dx = np.diff(g, axis=1)
    dy = np.diff(g, axis=0)
    return float(dx.var() + dy.var())


def generate_image(condition: dict, user_prompt: str, out_dir: Path, candidates: int, size: int) -> dict:
    import torch
    from diffusers import AutoPipelineForText2Image
    from torchvision.models import ResNet18_Weights, resnet18

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model_id = "stabilityai/sd-turbo"
    subject = (
        "a photorealistic close-up portrait of a healthy domestic tabby kitten, symmetrical face, anatomically correct ears, nose and whiskers"
        if user_prompt.strip() in {"고양이", "cat", "a cat"}
        else user_prompt.strip()
    )
    final_prompt = (
        f"{subject}, {condition['style']}, realistic individual fur strands, natural blue-gray eyes with catchlights, "
        "clean dark studio background, professional pet photography, coherent anatomy, highly detailed, sharp focus"
    )

    t0 = time.time()
    # fp16 repository variant minimizes download size, while float32 keeps CPU kernels reliable.
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing("max")
    pipe = pipe.to("cpu")
    load_s = time.time() - t0

    # Lightweight pretrained judge: choose the candidate most likely to be a cat.
    weights = ResNet18_Weights.DEFAULT
    judge = resnet18(weights=weights).eval().to("cpu")
    preprocess = weights.transforms()
    cat_indices = list(range(281, 286))  # tabby/tiger/Persian/Siamese/Egyptian cat in ImageNet-1k

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    best_score = -1e9
    infer_total = 0.0

    for k in range(candidates):
        seed = (int(condition["seed"]) + 104729 * k) % (2**31 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, 4, size // 8, size // 8), generator=gen, dtype=torch.float32)
        brain_map = torch.from_numpy(make_brain_latent(condition, k, size // 8, size // 8))[None]

        # Direct brain-state injection into the diffusion starting latent.
        alpha = 0.18 + 0.02 * (k % 3)
        latents = noise + alpha * brain_map
        latents = (latents - latents.mean()) / (latents.std() + 1e-6)

        t1 = time.time()
        image = pipe(
            prompt=final_prompt,
            generator=gen,
            latents=latents,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=size,
            width=size,
        ).images[0]
        infer_s = time.time() - t1
        infer_total += infer_s

        with torch.no_grad():
            logits = judge(preprocess(image).unsqueeze(0))
            probs = logits.softmax(dim=1)[0]
            cat_prob = float(probs[cat_indices].sum())
        sharp = sharpness_score(image)
        # Cat probability dominates; sharpness breaks ties.
        score = math.log(max(cat_prob, 1e-9)) + 0.00035 * sharp

        path = out_dir / f"candidate_{k+1:02d}.png"
        image.save(path)
        row = {
            "candidate": k + 1,
            "seed": seed,
            "brain_latent_alpha": alpha,
            "cat_probability": cat_prob,
            "sharpness": sharp,
            "selection_score": score,
            "inference_seconds": infer_s,
            "path": str(path),
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = image.copy()

    assert best is not None
    native_path = out_dir / "flybrain_sdturbo_v4_native.png"
    best.save(native_path)

    # Conservative final enlargement; generation itself remains native 512x512.
    final = best.resize((1024, 1024), Image.Resampling.LANCZOS)
    final = ImageEnhance.Contrast(final).enhance(1.04)
    final = final.filter(ImageFilter.UnsharpMask(radius=1.3, percent=125, threshold=2))
    final_path = out_dir / "flybrain_sdturbo_v4_cat.png"
    final.save(final_path)

    del pipe, judge
    gc.collect()

    return {
        "model_id": model_id,
        "final_prompt": final_prompt,
        "native_size": size,
        "final_size": 1024,
        "num_inference_steps": 1,
        "candidates": candidates,
        "direct_brain_latent_conditioning": True,
        "model_load_seconds": load_s,
        "total_inference_seconds": infer_total,
        "candidate_metrics": rows,
        "selected_candidate": int(max(rows, key=lambda x: x["selection_score"])["candidate"]),
        "native_path": str(native_path),
        "final_path": str(final_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly-dir", default="fly")
    ap.add_argument("--prompt", default="고양이")
    ap.add_argument("--brain-ms", type=float, default=180.0)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="output_v4")
    args = ap.parse_args()

    root = Path.cwd()
    condition = brain_condition((root / args.fly_dir).resolve(), args.prompt, args.brain_ms)
    os.chdir(root)
    generation = generate_image(condition, args.prompt, (root / args.out).resolve(), args.candidates, args.size)
    report = {
        "status": "completed",
        "architecture": "FlyWire-v783 whole-brain spikes -> direct latent adapter -> SD-Turbo -> ImageNet cat selector",
        "prompt": args.prompt,
        "brain": {k: v for k, v in condition.items() if not k.endswith("_hist")},
        "conditioning": {
            "neuron_hist_bins": 64,
            "time_hist_bins": 64,
            "active_id_hist_bins": 64,
            "direct_latent_injection": True,
        },
        "generation": generation,
    }
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_report_v4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
