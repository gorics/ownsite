import os, json, gzip, base64, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

OUT=Path('hybrid-result'); OUT.mkdir(exist_ok=True)
SEED=20260929
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
B64='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'
compact=json.loads(gzip.decompress(base64.b64decode(B64)).decode())
groups=compact['groups']; A=np.array(compact['A_scaled'],dtype=np.float32); sizes=np.array(compact['sizes'],dtype=np.float32)
G=len(groups)

sens=[i for i,g in enumerate(groups) if (g.startswith('VIS_') or g.startswith('OLF_') or g.startswith('MECH_') or g.startswith('THERMO_')) and sizes[i]>0]
drives=[i for i,g in enumerate(groups) if g.startswith('DRIVE_') and sizes[i]>0]
T=6000; X=np.zeros((T,G),np.float32); Y=np.zeros((T,G),np.float32); x=np.zeros(G,np.float32)
for t in range(T):
    stim=np.zeros(G,np.float32)
    if t%5==0 or np.random.rand()<.28:
        ids=np.random.choice(sens,np.random.randint(1,min(4,len(sens))+1),replace=False); stim[ids]=np.random.uniform(.2,1.1,len(ids))
    if drives and t%19==0:
        ids=np.random.choice(drives,1); stim[ids]=np.random.uniform(.1,.45,1)
    xn=np.clip(.88*x + .22*np.tanh(x@A) + stim,-2,2); X[t]=x; Y[t]=xn; x=xn
perm=np.random.permutation(T); tr=perm[:5400]; va=perm[5400:]
xt=torch.from_numpy(X[tr]); yt=torch.from_numpy(Y[tr]); xv=torch.from_numpy(X[va]); yv=torch.from_numpy(Y[va])
class BrainAdapter(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(G,64),nn.GELU(),nn.Linear(64,24),nn.Tanh()); self.dec=nn.Sequential(nn.Linear(24,64),nn.GELU(),nn.Linear(64,G))
    def forward(self,x):
        z=self.enc(x); return .88*x+self.dec(z),z
adapter=BrainAdapter(); opt=torch.optim.AdamW(adapter.parameters(),lr=2e-3)
for ep in range(20):
    p=torch.randperm(len(xt))
    for s in range(0,len(xt),256):
        b=p[s:s+256]; pred,_=adapter(xt[b]); loss=nn.functional.mse_loss(pred,yt[b]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    val=nn.functional.mse_loss(adapter(xv)[0],yv).item(); base=nn.functional.mse_loss(.88*xv,yv).item()
torch.save(adapter.state_dict(),OUT/'brain_adapter.pt')

state=np.zeros(G,np.float32)
def ix(n): return groups.index(n) if n in groups else None
for step in range(10):
    stim=np.zeros(G,np.float32)
    if step<4:
        for n,a in [('VIS_R1R6',.75),('VIS_ME',.65),('OLF_ORN_FOOD',1.0),('DRIVE_HUNGER',.35)]:
            j=ix(n)
            if j is not None: stim[j]=a
    state=np.clip(.88*state+.22*np.tanh(state@A)+stim,-2,2)
with torch.no_grad(): latent=adapter.enc(torch.from_numpy(state)).numpy()
top=np.argsort(np.abs(state))[-10:][::-1]
brain_summary=', '.join(f'{groups[i]}={state[i]:.3f}' for i in top)
latent_sig=float(np.mean(latent)); latent_energy=float(np.mean(latent**2))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
llm_id='HuggingFaceTB/SmolLM2-135M-Instruct'
tok=AutoTokenizer.from_pretrained(llm_id)
model=AutoModelForCausalLM.from_pretrained(llm_id)
model.config.use_cache=False
lora=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj'])
model=get_peft_model(model,lora); model.train()
train_texts=[]
for k in range(12):
    s=state.copy(); s += np.random.normal(0,.03,G).astype(np.float32)
    ti=np.argsort(np.abs(s))[-5:][::-1]
    labels=', '.join(groups[i] for i in ti)
    train_texts.append(f'FlyBrain state: {labels}.\nAssistant: Dominant connectome activity is concentrated in {labels}.')
optim=torch.optim.AdamW(model.parameters(),lr=7e-4)
for text in train_texts:
    batch=tok(text,return_tensors='pt',truncation=True,max_length=192)
    out=model(**batch,labels=batch['input_ids']); optim.zero_grad(); out.loss.backward(); optim.step()
model.eval(); model.config.use_cache=True
prompt=("You are the language head of a hybrid system controlled by a Drosophila connectome. "
        "Do not claim consciousness. Use the supplied neural state as control context.\n"
        f"Connectome state: {brain_summary}\nLatent signature={latent_sig:.4f}, energy={latent_energy:.4f}.\n"
        "User request: Describe what image this hybrid brain should generate: a fruit fly neural organism inside a glowing laboratory, highly detailed, cinematic, biological and computational elements merged.\nAssistant:")
inputs=tok(prompt,return_tensors='pt')
with torch.no_grad():
    gen=model.generate(**inputs,max_new_tokens=90,do_sample=True,temperature=float(np.clip(.55+abs(latent_sig)*.25,.55,.9)),top_p=.9,repetition_penalty=1.05)
text=tok.decode(gen[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
(OUT/'llm_output.txt').write_text(text,encoding='utf-8')
model.save_pretrained(OUT/'llm_brain_lora'); tok.save_pretrained(OUT/'llm_brain_lora')

class SDController(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(24,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,z): return self.net(z)
ctrl=SDController(); copt=torch.optim.AdamW(ctrl.parameters(),lr=3e-3)
with torch.no_grad():
    ztrain=adapter.enc(torch.from_numpy(X[np.random.choice(T,1200,replace=False)])).numpy()
C=[]
for z in ztrain:
    m=float(np.mean(z)); e=float(np.mean(z*z)); a=float(np.mean(np.abs(z)))
    C.append([2.5+4.0*min(1,e), 6+10*min(1,a), min(1,abs(m)*2.0)])
Z=torch.tensor(np.array(ztrain),dtype=torch.float32); C=torch.tensor(np.array(C),dtype=torch.float32)
for _ in range(120):
    pr=ctrl(Z); ls=nn.functional.mse_loss(pr,C); copt.zero_grad(); ls.backward(); copt.step()
with torch.no_grad(): c=ctrl(torch.from_numpy(latent)).numpy()
guidance=float(np.clip(c[0],2.0,7.0)); steps=int(np.clip(round(float(c[1])),6,18)); style=float(np.clip(c[2],0,1))
torch.save(ctrl.state_dict(),OUT/'sd_brain_controller.pt')

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
sd_id='segmind/tiny-sd'
pipe=DiffusionPipeline.from_pretrained(sd_id,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe=pipe.to('cpu')
seed=int((abs(latent_sig)*1e7 + latent_energy*1e8 + SEED))%(2**31-1)
style_tag='intricate neural filaments, bioluminescent synapses, cinematic macro photography' if style>.35 else 'scientific visualization, clean laboratory lighting'
img_prompt=('A biologically plausible fruit-fly neural organism fused with an AI machine in a glowing laboratory, '
            + style_tag + ', detailed Drosophila neural anatomy, visible computational graph motifs. ' + text[:220])
generator=torch.Generator(device='cpu').manual_seed(seed)
image=pipe(img_prompt,num_inference_steps=steps,guidance_scale=guidance,height=256,width=256,generator=generator).images[0]
image.save(OUT/'flybrain_hybrid.png')

summary={
 'seed':SEED,'connectome_groups':G,'adapter_val_mse':val,'baseline_mse':base,'improvement_factor':base/max(val,1e-12),
 'brain_top_state':brain_summary,'brain_latent_signature':latent_sig,'brain_latent_energy':latent_energy,
 'llm_model':llm_id,'llm_lora_train_examples':len(train_texts),'llm_output':text,
 'sd_model':sd_id,'sd_controller_train_samples':len(Z),'sd_seed':seed,'sd_guidance':guidance,'sd_steps':steps,'sd_style':style,
 'image_prompt':img_prompt
}
(OUT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
