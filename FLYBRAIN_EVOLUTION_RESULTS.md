# FlyBrain Open-Ended Evolution × LLM × Stable Diffusion

Branch: `flybrain-evolution-organism-20260929`

## Accepted structural champion

Parent FlyWire-derived connectome: 139,255 neurons / 2,698,236 synapses.

The latest accepted structural champion recorded by this project remains the result of the earlier validated local run (160 generations, population 72):

- baseline proxy fitness: -1.5839431329701001
- champion proxy fitness: 1.1906054767704406
- evolved neurons: 139,256 (+1 neo-neuron)
- evolved synapses: 2,686,587
- removed synapses: 17,465
- added synapses: 5,816
- full-LIF probe, food -> proboscis motor: 30 spikes
- full-LIF probe, touch -> abdomen motor: 24 spikes
- full-LIF probe, visual -> head motor: 10 spikes
- evolved connectome SHA256: `2f26abdbf31d26a6470c2379958779c848f1a614989d71ec8099bfa98f626a69`

Three later recorded descendants improved proxy fitness as high as 1.219447 but degraded full-LIF behavior. They remain rejected; the champion above is retained.

## Verified unified-organism GitHub Actions execution

Latest verified successful run at commit `bfd66aec448800a4708620b4ec6e7de8c2ba838f`: `36588656393`.
Artifact: `11042129668` (`flybrain-unified-organism-results`).

The run actually downloaded and used the FlyWire-derived FlyBrain connectome from `snedea/flybrain` and recorded:

- source neurons: 139,255
- source edges: 2,698,236
- source groups: 63
- compressed connectome SHA256: `fbf8d440ca1207c7573e1acdd2366f9d0beb9b533c1710f21681264f81b1cc49`
- LLM: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Stable Diffusion: `diffusers/tiny-stable-diffusion-torch`

CPU bounded training proof (2 optimizer steps):

- total loss: 5.9917893409729 -> 5.919310569763184
- LLM loss: 5.860219478607178 -> 5.78848123550415
- diffusion loss: 1.096401333808899 -> 1.0902230739593506
- generated image: 128x128 `sample.png`
- output checkpoint: `fusion_tissue.pt`
- seed: 20260929

The pretrained base LLM and diffusion organ weights are frozen in this CPU proof run; the shared fly/genome/adaptor tissue is trained. This is a single runnable parameter graph with differentiated organs, not a claim that Transformer/UNet tensors are biologically identical to Drosophila LIF neurons.

## Self-correction / acceptance gate

The current branch contains the unified-organism runner and its workflows, but it does **not** contain the earlier structural-evolution/full-LIF continuation implementation or a serialized copy of the accepted evolved connectome. Therefore a successful unified-organism Actions run is not, by itself, evidence of a new structural descendant or of improved full-LIF behavior. Earlier notes that implied the current branch workflow itself performed structural evolution plus full-LIF gating were incorrect.

Until structural mutation code, the accepted champion state, and the full-LIF validator are present and executed together, no new structural descendant may replace the accepted champion above. Unified-organism training runs may be reported as bounded training experiments only, and are not structural champion improvements.

## Limit

No finite evolutionary algorithm guarantees unbounded intelligence. Evolution can plateau or regress, hence objective validation and champion retention are required.