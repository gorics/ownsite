#!/usr/bin/env python3
"""Unified FlyBrain × SmolLM2 × tiny Stable Diffusion organism.

The fly connectome is compressed to a 63-area state vector. A single heritable latent is
injected into BOTH the LLM token stream and Stable-Diffusion text conditioning, and the
same shared tissue is optimized jointly against language and diffusion losses.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, struct, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline

LLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SD_ID = "diffusers/tiny-stable-diffusion-torch"

def load_fly(connectome: Path, meta: Path):
    raw_gz = connectome.read_bytes()
    source_sha256 = hashlib.sha256(raw_gz).hexdigest()
    raw = gzip.decompress(raw_gz)
    n, e = struct.unpack_from("<II", raw, 0)
    dt = np.dtype([("pre", "<u4"), ("post", "<u4"), ("w", "<f4")])
    ed = np.frombuffer(raw, dtype=dt, count=e, offset=8)
    mo = 8 + e * 12
    m = np.frombuffer(raw, dtype=np.uint8, count=n * 3, offset=mo).reshape(n, 3)
    group = (m[:, 1].astype(np.uint16) + (m[:, 2].astype(np.uint16) << 8)).astype(np.int32)
    meta_j = json.loads(meta.read_text())
    names = [x["name"] for x in meta_j["groups"]]
    g = len(names)
    W = np.zeros((g, g), np.float32)
    sg, dg = group[ed["pre"]], group[ed["post"]]
    np.add.at(W, (sg, dg), ed["w"].astype(np.float32))
    den = np.maximum(np.sum(np.abs(W), axis=1, keepdims=True), 1.0)
    W /= den
    return n, e, names, torch.from_numpy(W), source_sha256

def fly_dynamics(W: torch.Tensor, names, steps=18):
    name_to_i = {n:i for i,n in enumerate(names)}
    x = torch.zeros(1, W.shape[0])
    for name, val in [("OLF_ORN_FOOD",1.4),("VIS_ME",0.8),("MECH_BRISTLE",0.55),("DRIVE_HUNGER",0.35)]:
        if name in name_to_i:
            x[0, name_to_i[name]] = val
    traj=[]
    for _ in range(steps):
        x = torch.tanh(0.76*x + 1.55*(x @ W))
        traj.append(x)
    return torch.stack(traj, dim=1)

class UnifiedOrganism(nn.Module):
    def __init__(self, fly_dim, llm, llm_hidden, sd_text, sd_unet, sd_vae, sd_hidden, genome_dim=96):
        super().__init__()
        self.llm = llm
        self.sd_text = sd_text
        self.sd_unet = sd_unet
        self.sd_vae = sd_vae
        self.fly_encoder = nn.Sequential(nn.Linear(fly_dim,128), nn.Tanh(), nn.Linear(128,genome_dim), nn.Tanh())
        self.genome = nn.Parameter(torch.zeros(genome_dim))
        self.to_llm = nn.Sequential(nn.Linear(genome_dim,llm_hidden), nn.Tanh())
        self.to_sd = nn.Sequential(nn.Linear(genome_dim,sd_hidden), nn.Tanh())
        for module in [self.llm, self.sd_text, self.sd_unet, self.sd_vae]:
            for p in module.parameters(): p.requires_grad_(False)
    def latent(self, fly_state):
        return self.fly_encoder(fly_state) + self.genome.unsqueeze(0)

def text_loss(org, tok, fly_latent, text):
    ids = tok(text, return_tensors="pt").input_ids
    emb = org.llm.get_input_embeddings()(ids)
    bias = 0.035 * org.to_llm(fly_latent).unsqueeze(1)
    return org.llm(inputs_embeds=emb + bias, labels=ids).loss

def sd_noise_loss(org, pipe, fly_latent, prompt):
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids
    with torch.no_grad(): pe = org.sd_text(ids)[0]
    pe = pe + 0.035 * org.to_sd(fly_latent).unsqueeze(1)
    sample_size = int(org.sd_unet.config.sample_size)
    in_ch = int(org.sd_unet.config.in_channels)
    clean = torch.randn(1, in_ch, sample_size, sample_size)
    noise = torch.randn_like(clean)
    t = torch.tensor([int(pipe.scheduler.config.num_train_timesteps)//2], dtype=torch.long)
    noisy = pipe.scheduler.add_noise(clean, noise, t)
    pred = org.sd_unet(noisy, t, encoder_hidden_states=pe).sample
    return F.mse_loss(pred.float(), noise.float())

@torch.no_grad()
def generate_text(org, tok, fly_latent, prompt, max_new=28):
    ids = tok(prompt, return_tensors="pt").input_ids
    emb = org.llm.get_input_embeddings()(ids)
    fly_prefix = 0.08 * org.to_llm(fly_latent).unsqueeze(1)
    inp = torch.cat([fly_prefix, emb], dim=1)
    out = org.llm(inputs_embeds=inp, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen=[nxt]
    for _ in range(max_new-1):
        out = org.llm(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(nxt)
        if int(nxt.item()) == tok.eos_token_id: break
    return tok.decode(torch.cat(gen,dim=1)[0], skip_special_tokens=True)

@torch.no_grad()
def generate_image(org, pipe, fly_latent, prompt, out_path):
    def encode(s):
        ids = pipe.tokenizer(s, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        return org.sd_text(ids)[0]
    pe=encode(prompt); ne=encode("")
    pe=pe + 0.08*org.to_sd(fly_latent).unsqueeze(1)
    image=pipe(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
               num_inference_steps=4, guidance_scale=5.0).images[0]
    image.save(out_path)
    return image.size

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--connectome', default='data/connectome.bin.gz')
    ap.add_argument('--meta', default='data/neuron_meta.json')
    ap.add_argument('--out', default='results/unified')
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260929)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); started=time.time()
    n,e,names,W,source_sha256=load_fly(Path(a.connectome),Path(a.meta))
    fly_state=fly_dynamics(W,names)[:,-1,:]
    tok=AutoTokenizer.from_pretrained(LLM_ID)
    llm=AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32)
    pipe=StableDiffusionPipeline.from_pretrained(SD_ID, torch_dtype=torch.float32, safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    org=UnifiedOrganism(W.shape[0], llm, int(llm.config.hidden_size), pipe.text_encoder, pipe.unet, pipe.vae,
                        int(pipe.text_encoder.config.hidden_size))
    trainable=[p for p in org.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=2e-3); losses=[]
    train_text="Fly organism senses food, light, and touch; it integrates memory before acting."
    image_prompt="a microscopic fruit fly neural organism, glowing connectome, scientific visualization"
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); latent=org.latent(fly_state)
        l_lm=text_loss(org,tok,latent,train_text); l_sd=sd_noise_loss(org,pipe,latent,image_prompt)
        loss=l_lm + 0.12*l_sd + 1e-4*latent.square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        rec={'step':step,'total':float(loss.detach()),'llm':float(l_lm.detach()),'sd':float(l_sd.detach())}
        losses.append(rec); print(rec,flush=True)
    latent=org.latent(fly_state)
    text=generate_text(org,tok,latent,"The evolved fly-brain organism decides:")
    image_size=generate_image(org,pipe,latent,image_prompt,out/'sample.png')
    torch.save({'genome':org.genome.detach(),'fly_encoder':org.fly_encoder.state_dict(),
                'to_llm':org.to_llm.state_dict(),'to_sd':org.to_sd.state_dict()},out/'fusion_tissue.pt')
    result={'source_connectome':{'neurons':n,'edges':e,'groups':len(names),'compressed_sha256':source_sha256},
            'models':{'llm':LLM_ID,'stable_diffusion':SD_ID},
            'architecture':'single nn.Module organism; shared heritable latent injected into both LLM and SD conditioning',
            'base_organs_frozen':True,'trained_tissue':['fly_encoder','genome','to_llm','to_sd'],
            'training_losses':losses,'fly_state_l2':float(fly_state.norm()),'shared_latent_l2':float(latent.norm()),
            'generated_text':text,'generated_image':'sample.png','generated_image_size':list(image_size),
            'elapsed_sec':time.time()-started,'seed':a.seed}
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
