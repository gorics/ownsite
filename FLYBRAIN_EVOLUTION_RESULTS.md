# FlyBrain Open-Ended Evolution × LLM × Stable Diffusion

Branch: `flybrain-evolution-organism-20260929`

## Structural connectome evolution (validated local run)

Parent FlyWire-derived connectome: 139,255 neurons / 2,698,236 synapses.

160 generations, population 72:

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

Continuation eras use a proxy + exact LIF acceptance gate. Three tested descendants improved proxy fitness as high as 1.219447 but degraded full-LIF behavior, so they were rejected and the prior champion was retained.

## Unified model execution on GitHub Actions

Successful Actions run: `36572917028`
Artifact: `11035322380` (`flybrain-unified-organism-results`)

Actual open-source organs loaded by the workflow:

- FlyBrain connectome from `snedea/flybrain`
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `diffusers/tiny-stable-diffusion-torch`

`flybrain/unified_fly_llm_sd.py` instantiates a single PyTorch `nn.Module` organism containing the fly encoder, heritable genome state, LLM, Stable-Diffusion text encoder/UNet/VAE, and trainable fly-to-LLM/fly-to-SD tissues. The same heritable latent state is injected into both language and diffusion computation, and the fusion tissue is jointly optimized by language-modeling plus diffusion-noise losses.

CPU proof run (2 optimizer steps):

- total loss: 5.9917974472 -> 5.9193191528
- LLM loss: 5.8602275848 -> 5.7884898186
- diffusion loss: 1.0964013338 -> 1.0902230740
- generated image: 128x128 `sample.png`
- output checkpoint: `fusion_tissue.pt`
- exact output metadata: `result.json`

The pretrained base LLM and diffusion organ weights are frozen in this CPU proof run; the shared fly/genome/adaptor tissue is trained. This is a single runnable parameter graph with differentiated organs, not a claim that Transformer/UNet tensors are biologically identical to Drosophila LIF neurons.

## Limit

The evolutionary loop can run without an era limit in a persistent environment, but no finite algorithm guarantees unbounded intelligence. Evolution can plateau or regress, hence the full-LIF validation gate and champion retention.
