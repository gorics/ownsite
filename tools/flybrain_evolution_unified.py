#!/usr/bin/env python3
import gzip, json, math, os, random, struct, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

SEED=20260929
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT=Path(os.getenv('FLY_OUT','flybrain-evolution-results')); OUT.mkdir(parents=True,exist_ok=True)
FLY=Path(os.getenv('FLY_REPO','flybrain-src'))
GENS=int(os.getenv('GENERATIONS','40')); POP=int(os.getenv('POPULATION','24'))
SENS=[0,6,10,14]; MOTOR=np.array([56,57,58,35]); CENTRAL=np.array([17,19,21,23,25,26,27,28,35,60])
ACTIONS=['feed','orient','abdomen','descend']

def load_connectome():
    meta=json.loads((FLY/'data/neuron_meta.json').read_text()); names=[x['name'] for x in meta['groups']]
    with gzip.open(FLY/'data/connectome.bin.gz','rb') as f: raw=f.read()
    n,e=struct.unpack_from('<II',raw,0); ed=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')]); edges=np.frombuffer(raw,dtype=ed,count=e,offset=8)
    md=np.dtype([('region','u1'),('group','<u2')]); m=np.frombuffer(raw,dtype=md,count=n,offset=8+e*12); g=m['group'].astype(np.int64); G=len(names); B=np.zeros((G,G),np.float64)
    np.add.at(B,(g[edges['pre']],g[edges['post']]),edges['w'].astype(np.float64)); nz=np.abs(B[B!=0]); s=np.percentile(nz,90) if nz.size else 1.0
    B=np.tanh(B/max(s,1e-6)); B/=np.maximum(np.sum(np.abs(B),axis=1,keepdims=True),1.0); return n,e,names,B.astype(np.float32)

def init_genome(rng,G=63):
    return dict(scale=np.ones((G,G),np.float32),delta=np.zeros((G,G),np.float32),cap=np.ones(G,np.float32),exc=np.ones(G,np.float32),oi=rng.normal(0,.02,(16,G)).astype(np.float32),oo=rng.normal(0,.02,(G,16)).astype(np.float32),rr=rng.normal(0,.01,(16,16)).astype(np.float32),leak=.62,sigma=.12)
def clone(g): return {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in g.items()}
def mutate(g,rng,rate=.4):
    c=clone(g); G=len(c['cap']); c['sigma']=float(np.clip(c['sigma']*math.exp(rng.normal(0,.05)),.02,.6)); s=c['sigma']; k=max(3,int(G*G*rate*.012)); i=rng.integers(0,G,k); j=rng.integers(0,G,k)
    c['scale'][i,j]*=np.exp(rng.normal(0,s,k)).astype(np.float32); np.clip(c['scale'],0,4,out=c['scale'])
    for _ in range(max(5,int(7+18*rate))):
        a,b=int(rng.integers(G)),int(rng.integers(G)); r=rng.random()
        if r<.65: c['delta'][a,b]+=float(rng.normal(0,1.7*s))
        elif r<.9: c['delta'][a,b]*=float(rng.uniform(0,.4))
        else: c['delta'][a,b]=-c['delta'][a,b]+float(rng.normal(0,s))
    np.clip(c['delta'],-3,3,out=c['delta'])
    for _ in range(max(2,int(8*rate))):
        q=int(rng.integers(G)); c['cap'][q]*=float(math.exp(rng.normal(0,.6*s)))
        if q not in MOTOR and rng.random()<.07: c['cap'][q]*=float(rng.uniform(.03,.25))
    np.clip(c['cap'],.02,3.5,out=c['cap']); idx=rng.choice(G,max(2,int(G*.08)),replace=False); c['exc'][idx]*=np.exp(rng.normal(0,.35*s,len(idx))).astype(np.float32); np.clip(c['exc'],.25,3,out=c['exc'])
    for key,p,ss in [('oi',.03,.3),('oo',.03,.3),('rr',.05,.25)]:
        a=c[key]; mask=rng.random(a.shape)<p; a[mask]+=rng.normal(0,s*ss,int(mask.sum())).astype(np.float32); np.clip(a,-1,1,out=a)
    c['leak']=float(np.clip(c['leak']+rng.normal(0,.04*s),.2,.95)); return c
def cross(a,b,rng):
    c=clone(a); G=len(c['cap']); rows=rng.random(G)<.5; nodes=rng.random(G)<.5; c['scale'][rows]=b['scale'][rows]; c['delta'][rows]=b['delta'][rows]; c['cap'][nodes]=b['cap'][nodes]; c['exc'][nodes]=b['exc'][nodes]
    cols=rng.random(G)<.5; c['oi'][:,cols]=b['oi'][:,cols]; c['oo'][cols]=b['oo'][cols]; rm=rng.random(16)<.5; c['rr'][rm]=b['rr'][rm]
    if rng.random()<.5: c['leak']=b['leak']
    c['sigma']=(a['sigma']+b['sigma'])/2; return c
def eff(base,g):
    M=(base*g['scale']+.35*g['delta'])*np.sqrt(g['cap'][:,None]*g['cap'][None,:]); return (M/np.maximum(np.sum(np.abs(M),1,keepdims=True),1)).astype(np.float32)
def tasks(seed=SEED+11,n=48):
    rng=np.random.default_rng(seed); z=[]
    for k in range(n):
        y=k%4; cue=np.zeros(63,np.float32); cue[SENS[y]]=1; cue[SENS[(y+1)%4]]=.1; cue+=rng.normal(0,.025,63); l=np.zeros(8,np.float32); l[y]=1; l[4+(y+1)%4]=.4; l+=rng.normal(0,.025,8); s=np.zeros(8,np.float32); s[2*y]=1; s[2*y+1]=-.5; s+=rng.normal(0,.025,8); z.append((y,cue,l,s))
    return z
def score(g,base,ts):
    M=eff(base,g); ok=0; mar=[]; mem=[]; cm=[]; ene=[]
    for y,c,l,s in ts:
        b=np.zeros(63,np.float32); o=np.r_[l,s].astype(np.float32); early=None
        for t in range(10):
            ext=c if t<2 else 0; od=o@g['oi'] if 1<=t<=6 else 0; b=np.tanh(g['leak']*b+(b@M)*g['exc']+(ext+od)*g['exc']); o=np.tanh(.55*o+b@g['oo']+o@g['rr'])
            if t==2: early=b.copy()
        q=b[MOTOR]; ok+=int(np.argmax(q)==y); mar.append(float(q[y]-np.max(np.delete(q,y)))); mem.append(float(np.dot(early[CENTRAL],b[CENTRAL])/len(CENTRAL))); cm.append(float(np.dot(o,np.r_[l,s])/16)); ene.append(float(np.mean(np.abs(b))))
    acc=ok/len(ts); new=int(np.count_nonzero(np.abs(g['delta'])>1e-3)); low=int(np.count_nonzero(g['cap']<.08)); comp=new/(63*63)+.003*low; fit=5*acc+1.15*np.mean(mar)+.45*np.mean(mem)+.35*np.mean(cm)-.1*np.mean(ene)-.18*comp
    return float(fit),dict(accuracy=acc,margin=float(np.mean(mar)),memory=float(np.mean(mem)),crossmodal=float(np.mean(cm)),energy=float(np.mean(ene)),new_edges=new,low_capacity_groups=low,complexity=float(comp))
def evolve(base):
    rng=np.random.default_rng(SEED); ts=tasks(); root=init_genome(rng); pop=[root]+[mutate(root,rng,.7) for _ in range(POP-1)]; best=None; bs=-1e9; bm=None; hist=[]; stagn=0
    for gen in range(GENS+1):
        ev=[score(x,base,ts) for x in pop]; sc=np.array([x[0] for x in ev]); bi=int(sc.argmax())
        if sc[bi]>bs+1e-9: bs=float(sc[bi]); best=clone(pop[bi]); bm=ev[bi][1]; stagn=0
        else: stagn+=1
        row=dict(generation=gen,best_score=bs,mean_score=float(sc.mean()),**bm); hist.append(row)
        if gen%10==0 or gen==GENS: print('EVOLVE',json.dumps(row))
        if gen==GENS: break
        elite=max(2,POP//8); inds=np.argsort(sc)[-elite:][::-1]; nxt=[clone(pop[int(i)]) for i in inds]; rate=min(.9,.3+.015*stagn)
        def pick():
            q=rng.integers(0,POP,4); return int(q[np.argmax(sc[q])])
        while len(nxt)<POP: nxt.append(mutate(cross(pop[pick()],pop[pick()],rng),rng,rate))
        pop=nxt
    return best,bs,bm,hist
def compress8(t):
    x=t.detach().float().cpu().flatten().numpy(); p=np.array_split(x,8); z=np.array([a.mean() for a in p],np.float32); z=(z-z.mean())/(z.std()+1e-6); return np.tanh(z)
def train_llm(summary):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid='sshleifer/tiny-gpt2'; tok=AutoTokenizer.from_pretrained(mid); tok.pad_token=tok.eos_token; model=AutoModelForCausalLM.from_pretrained(mid); model.train(); samples=[f'FlyBrain {name}; {summary}; ACTION: {act}; IMAGE: macro fruit fly {name}' for name,act in zip(['food','visual','mechanical','thermal'],ACTIONS)]
    batch=tok(samples,return_tensors='pt',padding=True,truncation=True,max_length=96); labels=batch.input_ids.clone(); labels[batch.attention_mask==0]=-100; opt=torch.optim.AdamW(model.parameters(),lr=2e-4); losses=[]
    with torch.no_grad(): before=float(model(**batch,labels=labels).loss)
    for i in range(12):
        loss=model(**batch,labels=labels).loss; opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss)); print('LLM_TRAIN',i,float(loss))
    model.eval()
    with torch.no_grad():
        after=float(model(**batch,labels=labels).loss); inp=tok(f'FlyBrain food; {summary}; ACTION:',return_tensors='pt'); out=model.generate(**inp,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id); text=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip(); hid=model(**inp,output_hidden_states=True).hidden_states[-1].mean((0,1))
    (OUT/'llm_output.txt').write_text(text); return compress8(hid),dict(model=mid,loss_before=before,loss_after=after,losses=losses,generated=text)
def target(action,size=64):
    bg=[(210,240,190),(185,215,245),(245,205,185),(220,200,240)][ACTIONS.index(action)]; im=Image.new('RGB',(size,size),bg); d=ImageDraw.Draw(im); d.ellipse((22,22,42,46),fill=(55,40,30)); d.ellipse((26,13,38,27),fill=(45,35,30)); d.ellipse((12,20,28,36),fill=(235,235,245)); d.ellipse((36,20,52,36),fill=(235,235,245)); return im
def train_sd(action,summary):
    from diffusers import StableDiffusionPipeline
    mid='diffusers/tiny-stable-diffusion-torch'; pipe=StableDiffusionPipeline.from_pretrained(mid,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False); pipe.set_progress_bar_config(disable=True); pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False); pipe.unet.train(); prompt=f'macro fruit fly {action}, evolved connectome, {summary[:80]}'; im=target(action); im.save(OUT/'sd_training_target.png'); x=torch.from_numpy(np.asarray(im).astype(np.float32)/127.5-1).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        lat=pipe.vae.encode(x).latent_dist.sample()*float(getattr(pipe.vae.config,'scaling_factor',0.18215)); ti=pipe.tokenizer([prompt],padding='max_length',max_length=pipe.tokenizer.model_max_length,truncation=True,return_tensors='pt'); emb=pipe.text_encoder(ti.input_ids)[0]
    opt=torch.optim.AdamW(pipe.unet.parameters(),lr=1e-4); losses=[]
    for i in range(4):
        noise=torch.randn_like(lat); t=torch.randint(0,pipe.scheduler.config.num_train_timesteps,(1,),dtype=torch.long); noisy=pipe.scheduler.add_noise(lat,noise,t); pred=pipe.unet(noisy,t,encoder_hidden_states=emb).sample; loss=F.mse_loss(pred.float(),noise.float()); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss)); print('SD_TRAIN',i,float(loss))
    pipe.unet.eval(); gen=torch.Generator().manual_seed(SEED+77); out=pipe(prompt,num_inference_steps=4,guidance_scale=1.0,height=64,width=64,generator=gen).images[0]; p=OUT/'stable_diffusion_output.png'; out.save(p); a=np.asarray(out).astype(np.float32)/255.; gray=a.mean(2); feat=np.array([a[:,:,0].mean(),a[:,:,1].mean(),a[:,:,2].mean(),a.std(),gray[:32,:32].mean(),gray[:32,32:].mean(),gray[32:,:32].mean(),gray[32:,32:].mean()],np.float32); feat=(feat-feat.mean())/(feat.std()+1e-6); return np.tanh(feat),dict(model=mid,losses=losses,prompt=prompt,image_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fused(g,base,lf,sf):
    M=eff(base,g); b=np.zeros(63,np.float32); o=np.r_[lf,sf].astype(np.float32); c=np.zeros(63,np.float32); c[SENS[0]]=1; tr=[]
    for t in range(14):
        ext=c if t<2 else 0; b=np.tanh(g['leak']*b+(b@M)*g['exc']+(ext+o@g['oi'])*g['exc']); o=np.tanh(.55*o+b@g['oo']+o@g['rr']); tr.append(dict(tick=t,brain_energy=float(np.mean(np.abs(b))),organ_energy=float(np.mean(np.abs(o)))))
    q=b[MOTOR]; return ACTIONS[int(np.argmax(q))],q.tolist(),o.tolist(),tr
def main():
    t=time.time(); n,e,names,base=load_connectome(); print('CONNECTOME',n,e,len(names)); g,bs,bm,hist=evolve(base); np.savez_compressed(OUT/'best_genome.npz',**{k:v for k,v in g.items() if isinstance(v,np.ndarray)},leak=np.array([g['leak']]),sigma=np.array([g['sigma']])); (OUT/'evolution_history.json').write_text(json.dumps(hist,indent=2)); summary=f'score {bs:.4f}; accuracy {bm["accuracy"]:.3f}; new_edges {bm["new_edges"]}'; lf,lm=train_llm(summary); pre,_,_,_=fused(g,base,lf,np.zeros(8,np.float32)); sf,sm=train_sd(pre,summary); act,logits,org,tr=fused(g,base,lf,sf)
    result=dict(status='ok',connectome=dict(neurons=n,edges=e,groups=len(names)),evolution=dict(generations=GENS,population=POP,score=bs,metrics=bm,baseline=hist[0],final=hist[-1]),llm=lm,stable_diffusion=sm,fusion=dict(provisional_action=pre,final_action=act,motor_logits=logits,final_organ_state=org,trace=tr),runtime_sec=time.time()-t); (OUT/'unified_result.json').write_text(json.dumps(result,indent=2)); print('FINAL_JSON='+json.dumps(result))
if __name__=='__main__': main()
