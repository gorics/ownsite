#!/usr/bin/env python3
import os, json, gzip, struct, time, hashlib, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=20260929
np.random.seed(SEED); torch.manual_seed(SEED)
OUT=Path(os.environ.get('FLY_OUT','flybrain-real-results')); OUT.mkdir(parents=True,exist_ok=True)

class FlyConnectome:
    def __init__(self,path):
        with gzip.open(path,'rb') as f: raw=f.read()
        self.n,self.e=struct.unpack_from('<II',raw,0)
        ed=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')])
        edges=np.frombuffer(raw,dtype=ed,count=self.e,offset=8)
        mo=8+self.e*12; md=np.dtype([('region','u1'),('group','<u2')])
        meta=np.frombuffer(raw,dtype=md,count=self.n,offset=mo)
        self.pre=edges['pre']; self.post=edges['post']; self.w=edges['w'].astype(np.float32).copy()
        self.w*=0.15/max(float(np.max(np.abs(self.w))),1e-8)
        c=np.bincount(self.pre,minlength=self.n).astype(np.uint32); self.row=np.empty(self.n+1,np.uint32); self.row[0]=0; np.cumsum(c,out=self.row[1:])
        self.g=meta['group'].astype(np.int32,copy=False); self.ng=int(self.g.max())+1
        self.gidx=[np.flatnonzero(self.g==i) for i in range(self.ng)]
    def run(self,groups,ticks=24):
        V=np.zeros(self.n,np.float32); fired=np.zeros(self.n,bool); ref=np.zeros(self.n,np.uint8)
        stim=np.concatenate([self.gidx[x] for x in groups if x < self.ng and len(self.gidx[x])])
        total=np.zeros(self.ng,np.float32)
        for _ in range(ticks):
            rp=ref>0; ref[rp]-=1; V[rp]=0; V[~rp]*=.95
            for i in np.flatnonzero(fired):
                s,e=int(self.row[i]),int(self.row[i+1])
                if e>s: np.add.at(V,self.post[s:e],self.w[s:e])
            ok=stim[ref[stim]==0]; V[ok]+=1.2
            fired[:]=False; can=(ref==0)&(V>=1.0); fired[can]=True; V[can]=0; ref[can]=3
            total+=np.bincount(self.g[can],minlength=self.ng)
        feat=np.log1p(total); feat/=np.linalg.norm(feat)+1e-6
        return feat,total,int(stim.size)

class Readout(nn.Module):
    def __init__(self,d,k): super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,k))
    def forward(self,x): return self.net(x)

def train_readout(features):
    X=[]; y=[]
    for i,f in enumerate(features):
        for _ in range(128):
            z=f+np.random.normal(0,.02,f.shape).astype(np.float32); z/=np.linalg.norm(z)+1e-6; X.append(z); y.append(i)
    X=torch.tensor(np.stack(X)); y=torch.tensor(y)
    m=Readout(X.shape[1],len(features)); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    for _ in range(350):
        q=torch.randint(0,len(X),(64,)); loss=F.cross_entropy(m(X[q]),y[q]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): acc=float((m(X).argmax(1)==y).float().mean())
    return m,acc

def main():
    t0=time.time(); repo=Path(os.environ.get('FLY_REPO','flybrain-src'))
    data=repo/'data'; meta=json.loads((data/'neuron_meta.json').read_text())
    names=[f'GROUP_{i}' for i in range(meta['group_count'])]
    for x in meta['groups']: names[x['id']]=x['name']
    fly=FlyConnectome(data/'connectome.bin.gz')
    modes={'food_reward':[6,32],'danger':[7,33],'visual_explore':[0,2],'mechanical_startle':[10,11]}
    states={}
    for mode,gs in modes.items():
        feat,total,nstim=fly.run(gs)
        top=np.argsort(total)[::-1][:6]
        states[mode]={'feat':feat,'total':total,'stim_neurons':nstim,'top':[(names[int(i)],int(total[i])) for i in top if total[i]>0]}
        print('STATE',mode,states[mode]['top'])
    features=np.stack([states[m]['feat'] for m in modes]); readout,acc=train_readout(features)
    mixed=features[0]+features[2]; mixed/=np.linalg.norm(mixed)+1e-6
    with torch.no_grad(): probs=F.softmax(readout(torch.tensor(mixed).unsqueeze(0)),1)[0].numpy()
    mode_names=list(modes); selected=mode_names[int(probs.argmax())]
    brain_top=states[selected]['top'][:5]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    llm_id='HuggingFaceTB/SmolLM2-135M-Instruct'
    tok=AutoTokenizer.from_pretrained(llm_id)
    llm=AutoModelForCausalLM.from_pretrained(llm_id,torch_dtype=torch.float32)
    sys='You are the language readout attached to a fruit-fly connectome. Return exactly two lines: STATE: one concise sentence; IMAGE_PROMPT: one vivid safe English text-to-image prompt.'
    usr=f"Connectome classification={selected}; probabilities={dict(zip(mode_names,[round(float(x),4) for x in probs]))}; top active groups={brain_top}."
    text=tok.apply_chat_template([{'role':'system','content':sys},{'role':'user','content':usr}],tokenize=False,add_generation_prompt=True)
    inp=tok(text,return_tensors='pt')
    with torch.no_grad(): out=llm.generate(**inp,max_new_tokens=100,do_sample=True,temperature=.7,top_p=.9,repetition_penalty=1.08)
    llm_text=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
    print('LLM_OUTPUT\n'+llm_text)
    m=re.search(r'IMAGE_PROMPT\s*:\s*(.+)',llm_text,re.I)
    prompt=(m.group(1).strip() if m else 'macro photograph of a fruit fly exploring luminous leaves, detailed natural light')
    del llm; import gc; gc.collect()

    from diffusers import DiffusionPipeline
    sd_id='segmind/tiny-sd'
    pipe=DiffusionPipeline.from_pretrained(sd_id,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
    pipe.enable_attention_slicing()
    gen=torch.Generator(device='cpu').manual_seed(SEED + int(np.argmax(probs))*101)
    image=pipe(prompt,num_inference_steps=8,guidance_scale=5.5,height=256,width=256,generator=gen).images[0]
    img_path=OUT/'stable_diffusion_output.png'; image.save(img_path)
    sha=hashlib.sha256(img_path.read_bytes()).hexdigest()

    result={
      'status':'ok','connectome':{'neurons':fly.n,'edges':fly.e},'modes':{},
      'readout_train_accuracy':acc,'mixed_probs':dict(zip(mode_names,[float(x) for x in probs])),'selected_mode':selected,
      'llm_model':llm_id,'llm_output':llm_text,'stable_diffusion_model':sd_id,'image_prompt':prompt,'image_sha256':sha,
      'runtime_sec':time.time()-t0
    }
    for k,v in states.items(): result['modes'][k]={'stim_neurons':v['stim_neurons'],'total_spikes':int(v['total'].sum()),'top_groups':v['top']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'llm_output.txt').write_text(llm_text+'\n\nIMAGE_PROMPT: '+prompt+'\n',encoding='utf-8')
    (OUT/'README.txt').write_text('Actual run: snedea/flybrain connectome -> trained readout adapter -> SmolLM2-135M-Instruct -> segmind/tiny-sd.\n',encoding='utf-8')
    print('FINAL_JSON='+json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
