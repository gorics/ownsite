#!/usr/bin/env python3
import numpy as np
import torch
from pathlib import Path
import flybrain_evolution_unified as core


def safe_compress8(t):
    x=t.detach().float().cpu().flatten().numpy().astype(np.float32)
    if x.size==0:
        return np.zeros(8,np.float32)
    if x.size<8:
        xp=np.arange(x.size,dtype=np.float32)
        x=np.interp(np.linspace(0,max(x.size-1,1),8),xp,x).astype(np.float32)
        z=x
    else:
        z=np.array([p.mean() for p in np.array_split(x,8)],np.float32)
    z=np.nan_to_num(z,nan=0.0,posinf=6.0,neginf=-6.0)
    z=(z-z.mean())/(z.std()+1e-6)
    return np.tanh(z)


def train_llm_smollm(summary):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid='HuggingFaceTB/SmolLM2-135M-Instruct'
    tok=AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad=False
    trainable=[]
    if hasattr(model,'model') and hasattr(model.model,'layers'):
        for layer in model.model.layers[-2:]:
            for p in layer.parameters(): p.requires_grad=True; trainable.append(p)
    for p in model.lm_head.parameters():
        p.requires_grad=True; trainable.append(p)
    samples=[]
    for state,action in zip(['food reward','visual exploration','mechanical startle','thermal state'],core.ACTIONS):
        samples.append(f'Fruit-fly connectome state: {state}. {summary}.\nACTION: {action}\nIMAGE_PROMPT: macro photograph of a fruit fly, {state}, evolved neural organism')
    batch=tok(samples,return_tensors='pt',padding=True,truncation=True,max_length=128)
    labels=batch.input_ids.clone(); labels[batch.attention_mask==0]=-100
    model.train(); opt=torch.optim.AdamW(trainable,lr=5e-5); losses=[]
    with torch.no_grad(): before=float(model(**batch,labels=labels).loss)
    for i in range(10):
        loss=model(**batch,labels=labels).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        losses.append(float(loss.detach())); print('SMOLLM_TRAIN',i,losses[-1])
    model.eval(); prompt=f'Fruit-fly connectome state: food reward. {summary}.\nACTION:'
    inp=tok(prompt,return_tensors='pt')
    with torch.no_grad():
        after=float(model(**batch,labels=labels).loss)
        out=model.generate(**inp,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
        hid=model(**inp,output_hidden_states=True).hidden_states[-1].mean(dim=(0,1))
    (core.OUT/'llm_output.txt').write_text(text,encoding='utf-8')
    feat=safe_compress8(hid)
    assert np.isfinite(feat).all()
    return feat,dict(model=mid,loss_before=before,loss_after=after,losses=losses,generated=text,latent=feat.tolist())


core.compress8=safe_compress8
core.train_llm=train_llm_smollm
core.main()
