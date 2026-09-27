from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


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
    spikes_t = np.rint(np.asarray(trial.spikes_t_ms, dtype=np.float64) * 1000.0).astype(np.int32)
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(np.asarray([brain.n_neurons, brain.n_synapses], dtype=np.int64).tobytes())
    h.update(spikes_i.tobytes())
    h.update(spikes_t.tobytes())
    if spikes_i.size:
        active = np.unique(spikes_i)
        h.update(np.asarray(brain.i2flyid[active], dtype=np.int64).tobytes())
    digest = h.digest()
    seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)

    light = ["soft natural window light", "cinematic rim lighting", "golden-hour sunlight", "cool overcast daylight", "studio softbox lighting", "dramatic side lighting"][digest[8] % 6]
    lens = ["85mm portrait lens", "50mm prime lens", "shallow depth of field", "crisp documentary photography", "macro-level fur detail"][digest[9] % 5]
    palette = ["natural colors", "warm neutral palette", "cool neutral palette", "rich but realistic colors"][digest[10] % 4]

    return {
        "seed": seed,
        "digest": digest.hex(),
        "style": f"{light}, {lens}, {palette}",
        "n_neurons": int(brain.n_neurons),
        "n_synapses": int(brain.n_synapses),
        "build_seconds": build_s,
        "trial_sim_ms": float(trial.sim_ms),
        "trial_wall_ms": float(trial.wall_ms),
        "active_neurons": int(trial.active_neurons),
        "spike_count": int(spikes_i.size),
        "mn9_spikes": int(trial.mn9_spikes),
        "eat": bool(trial.extra.get("eat", False)),
    }


def generate_image(condition: dict, user_prompt: str, out_dir: Path, steps: int, size: int) -> dict:
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model_id = "segmind/tiny-sd"
    subject = "a photorealistic portrait photo of a beautiful domestic cat" if user_prompt.strip() in {"고양이", "cat", "a cat"} else user_prompt.strip()
    final_prompt = f"{subject}, {condition['style']}, highly detailed fur, realistic eyes, professional photography, coherent anatomy, sharp focus"
    negative = "blurry, low resolution, low quality, deformed, malformed, duplicate, extra limbs, bad anatomy, text, watermark, logo, oversaturated, cartoon"

    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cpu")
    load_s = time.time() - t0

    generator = torch.Generator(device="cpu").manual_seed(int(condition["seed"]))
    t1 = time.time()
    image = pipe(prompt=final_prompt, negative_prompt=negative, generator=generator, num_inference_steps=steps, guidance_scale=7.0, height=size, width=size).images[0]
    infer_s = time.time() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    native_path = out_dir / "flybrain_tinysd_native.png"
    final_path = out_dir / "flybrain_tinysd_cat.png"
    image.save(native_path)
    image.resize((768, 768), resample=3).save(final_path)
    return {"model_id": model_id, "final_prompt": final_prompt, "negative_prompt": negative, "steps": steps, "native_size": size, "model_load_seconds": load_s, "inference_seconds": infer_s, "native_path": str(native_path), "final_path": str(final_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly-dir", default="fly")
    ap.add_argument("--prompt", default="고양이")
    ap.add_argument("--brain-ms", type=float, default=150.0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    root = Path.cwd()
    condition = brain_condition((root / args.fly_dir).resolve(), args.prompt, args.brain_ms)
    os.chdir(root)
    generation = generate_image(condition, args.prompt, (root / args.out).resolve(), args.steps, args.size)
    report = {"status": "completed", "architecture": "FlyWire-v783 whole-brain spike conditioning -> Tiny-SD diffusion", "prompt": args.prompt, "brain": condition, "generation": generation}
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# trigger: public workflow registered on main
