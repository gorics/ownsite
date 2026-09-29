#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flybrain_evolution_unified as core


def safe_compress8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().flatten().numpy().astype(np.float32)
    if x.size == 0:
        return np.zeros(8, dtype=np.float32)
    if x.size < 8:
        xp = np.arange(x.size, dtype=np.float32)
        z = np.interp(np.linspace(0, max(x.size - 1, 1), 8), xp, x).astype(np.float32)
    else:
        z = np.array([part.mean() for part in np.array_split(x, 8)], dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-6.0)
    z = (z - z.mean()) / (z.std() + 1e-6)
    return np.tanh(z)


def train_llm_organ(summary: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    trainable_names = []
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError("SmolLM2 layer layout not found")
    for li, layer in enumerate(model.model.layers[-2:], start=len(model.model.layers) - 2):
        for name, p in layer.named_parameters():
            p.requires_grad = True
            trainable.append(p)
            trainable_names.append(f"model.layers.{li}.{name}")

    samples = []
    for state, action in zip(
        ["food reward", "visual exploration", "mechanical startle", "thermal state"],
        core.ACTIONS,
    ):
        samples.append(
            f"Fruit-fly connectome state: {state}. {summary}.\n"
            f"Required motor response: {action}.\n"
            f"ACTION: {action}\n"
            f"IMAGE_PROMPT: macro biological photograph of a fruit fly showing {state}."
        )

    batch = tok(samples, return_tensors="pt", padding=True, truncation=True, max_length=144)
    labels = batch.input_ids.clone()
    labels[batch.attention_mask == 0] = -100

    model.train()
    opt = torch.optim.AdamW(trainable, lr=8e-5)
    with torch.no_grad():
        before = float(model(**batch, labels=labels).loss)
    losses = []
    for step in range(14):
        loss = model(**batch, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        print("SMOLLM_ORGAN_TRAIN", step, losses[-1])

    model.eval()
    with torch.no_grad():
        after = float(model(**batch, labels=labels).loss)
        prompt = f"Fruit-fly connectome state: food reward. {summary}.\nACTION:"
        inp = tok(prompt, return_tensors="pt")
        out = model.generate(
            **inp,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        hidden = model(**inp, output_hidden_states=True).hidden_states[-1].mean(dim=(0, 1))

    feat = safe_compress8(hidden)
    if not np.isfinite(feat).all():
        raise RuntimeError("LLM organ produced non-finite latent")

    # Persist the actually trained neural tissue, not just metrics.
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu()
    llm_state_path = core.OUT / "llm_organ_state.pt"
    torch.save(state, llm_state_path)
    (core.OUT / "llm_output.txt").write_text(text, encoding="utf-8")

    return feat, {
        "model": model_id,
        "loss_before": before,
        "loss_after": after,
        "losses": losses,
        "generated": text,
        "latent": feat.tolist(),
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "saved_state_bytes": llm_state_path.stat().st_size,
        "saved_trainable_tensors": len(state),
    }


def train_sd_organ(action: str, summary: str):
    from diffusers import StableDiffusionPipeline

    model_id = "segmind/tiny-sd"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.conv_out.requires_grad_(True)
    pipe.unet.train()

    initial = {k: v.detach().cpu().clone() for k, v in pipe.unet.conv_out.state_dict().items()}
    trainable = list(pipe.unet.conv_out.parameters())

    prompt = (
        "macro biological photograph of a Drosophila fruit fly on a green leaf, "
        f"motor state {action}, evolved connectome, detailed transparent wings, compound eyes, "
        f"natural light, high detail; {summary[:90]}"
    )
    target = core.target(action, size=256)
    target.save(core.OUT / "sd_training_target.png")
    x = torch.from_numpy(np.asarray(target).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        lat = pipe.vae.encode(x).latent_dist.sample()
        lat = lat * float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
        ti = pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        emb = pipe.text_encoder(ti.input_ids)[0]
        generator = torch.Generator(device="cpu").manual_seed(core.SEED + 301)
        fixed_noise = torch.randn(lat.shape, generator=generator, dtype=lat.dtype)
        timestep = torch.tensor([min(500, pipe.scheduler.config.num_train_timesteps - 1)], dtype=torch.long)
        noisy = pipe.scheduler.add_noise(lat, fixed_noise, timestep)

    opt = torch.optim.AdamW(trainable, lr=1e-4)
    losses = []
    for step in range(8):
        pred = pipe.unet(noisy, timestep, encoder_hidden_states=emb).sample
        loss = F.mse_loss(pred.float(), fixed_noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        print("TINY_SD_ORGAN_TRAIN", step, losses[-1])

    pipe.unet.eval()
    delta = {}
    sq = 0.0
    for k, v in pipe.unet.conv_out.state_dict().items():
        d = v.detach().cpu() - initial[k]
        delta[k] = d
        sq += float((d.float() ** 2).sum())
    sd_delta_path = core.OUT / "sd_organ_delta.pt"
    torch.save(delta, sd_delta_path)

    gen = torch.Generator(device="cpu").manual_seed(core.SEED + 777)
    image = pipe(
        prompt,
        num_inference_steps=12,
        guidance_scale=7.0,
        height=256,
        width=256,
        generator=gen,
    ).images[0]
    image_path = core.OUT / "stable_diffusion_output.png"
    image.save(image_path)

    arr = np.asarray(image).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    h, w = gray.shape
    feat = np.array(
        [
            arr[:, :, 0].mean(),
            arr[:, :, 1].mean(),
            arr[:, :, 2].mean(),
            arr.std(),
            gray[: h // 2, : w // 2].mean(),
            gray[: h // 2, w // 2 :].mean(),
            gray[h // 2 :, : w // 2].mean(),
            gray[h // 2 :, w // 2 :].mean(),
        ],
        dtype=np.float32,
    )
    feat = np.nan_to_num(feat)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    feat = np.tanh(feat)
    if not np.isfinite(feat).all():
        raise RuntimeError("Diffusion organ produced non-finite latent")

    return feat, {
        "model": model_id,
        "losses": losses,
        "prompt": prompt,
        "image_width": image.width,
        "image_height": image.height,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "trained_delta_l2": sq ** 0.5,
        "saved_delta_bytes": sd_delta_path.stat().st_size,
        "latent": feat.tolist(),
    }


core.compress8 = safe_compress8
core.train_llm = train_llm_organ
core.train_sd = train_sd_organ

if __name__ == "__main__":
    core.main()
