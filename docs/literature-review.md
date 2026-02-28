# Literature Review: Oscillatory Dynamics for Silence-Robust ASR

## Context

Pulse-Whisper injects learnable oscillatory signals into Whisper's encoder to reduce hallucinations during silence/noise gaps. Phase 1 experiments showed that encoder-only pulse injection on a frozen model degrades WER at every constraint level tested. This review surveys relevant work to inform a novel architecture that preserves the oscillatory mechanism while addressing the frozen decoder bottleneck.

---

## 1. Oscillatory Mechanisms in Neural Networks

### 1.1 Artificial Kuramoto Oscillatory Neurons (AKOrN)
- **Authors**: Miyato et al.
- **Venue**: ICLR 2025
- **Links**: [arXiv](https://arxiv.org/abs/2410.13821), [Project](https://takerum.github.io/akorn_project_page/)
- **Mechanism**: Replaces threshold activations with N-dimensional oscillatory neurons governed by a generalized Kuramoto model. Neurons synchronize phases to bind related features — a continuous, distributed clustering mechanism. When implemented in self-attention, AKOrN combines transformer clustering with oscillatory synchronization.
- **Results**: Strong on unsupervised object discovery, adversarial robustness, calibrated uncertainty, and reasoning.
- **Relevance**: Closest existing work to pulse injection. Shows oscillatory dynamics in transformer layers improve robustness and feature binding. Key difference: AKOrN replaces activations entirely; we inject additively. Their synchronization-as-binding idea could inspire pulse phases that synchronize across layers to create coherent "silence" vs "speech" representations.
- **Limitation**: Does not address encoder-decoder architectures or frozen components.

### 1.2 KomplexNet: Complex-Valued Kuramoto
- **Links**: [arXiv](https://arxiv.org/abs/2502.21077), [OpenReview](https://openreview.net/forum?id=zx6QGmBL43)
- **Mechanism**: Complex-valued neurons where amplitude retains standard CNN functionality and phase encodes binding through Kuramoto dynamics. Two variants: feedforward and recurrent (with top-down feedback for phase refinement).
- **Relevance**: The amplitude/phase decomposition maps directly to our pulse equation. Amplitude A is analogous to their magnitude channel, phi(h) to their phase channel. The recurrent feedback variant could inspire a decoder-to-encoder phase feedback mechanism — decoder state influencing encoder pulse phase without unfreezing decoder weights.

### 1.3 Coupled Oscillatory Recurrent Neural Network (coRNN)
- **Authors**: Rusch, Mishra
- **Venue**: ICLR 2021 (Oral)
- **Links**: [arXiv](https://arxiv.org/abs/2010.00951), [GitHub](https://github.com/tk-rusch/coRNN)
- **Mechanism**: RNN based on time-discretized second-order ODEs of coupled nonlinear oscillators. Provides precise gradient bounds, mitigating exploding/vanishing gradients. System: `y'' + gamma*y' + tanh(W*y + b) = 0` where gamma controls damping.
- **Relevance**: Foundational theory for oscillatory neural dynamics. Our pulse equation is a first-order additive oscillation; coRNN shows oscillatory ODEs maintain gradient stability over long sequences. Damping parameter gamma is analogous to our alpha. Could frame pulse injection as coRNN-style oscillatory residual.

### 1.4 SIREN: Sinusoidal Representation Networks
- **Authors**: Sitzmann et al.
- **Venue**: NeurIPS 2020
- **Links**: [arXiv](https://arxiv.org/abs/2006.09661), [Project](https://www.vincentsitzmann.com/siren/)
- **Mechanism**: Uses sin(x) as activation function. Any derivative of a SIREN is itself a SIREN. Dramatically better at representing continuous signals and their derivatives.
- **Relevance**: Proves sinusoidal functions are universal approximators for continuous signals. Our sin()-based pulse inherits this theoretical backing. The derivative-preservation property means gradients through our pulse module are well-behaved.

---

## 2. Whisper Hallucination Reduction

### 2.1 Calm-Whisper (Critical Finding)
- **Title**: "Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down"
- **Authors**: Wang et al.
- **Venue**: Interspeech 2025
- **Link**: [arXiv](https://arxiv.org/abs/2505.12969)
- **Mechanism**: Identifies that only **3 of 20 decoder self-attention heads** (heads #1, #6, #11) cause over 75% of hallucinations on non-speech audio. Fine-tunes ONLY these 3 heads on noise data with blank labels, freezing everything else.
- **Results**: Over 80% hallucination reduction with less than 0.1% WER degradation on LibriSpeech.
- **Relevance**: **Game-changing for our project.** Demonstrates that:
  1. The hallucination problem is localized to specific decoder heads
  2. Targeted decoder-side intervention works
  3. Minimal parameter changes suffice
- **Implication**: Our encoder-only pulse injection was targeting the wrong location. Pulse should target decoder heads #1, #6, #11 — inject oscillatory signals that regularize their attention during silence without modifying their weights.

### 2.2 Investigation of Whisper Hallucinations (ICASSP 2025)
- **Authors**: AGH University researchers
- **Link**: [arXiv](https://arxiv.org/abs/2501.11378)
- **Mechanism**: Creates "Bag of Hallucinations" (BoH) — curated set of frequently-hallucinated phrases. Mitigation strategies: beam_size=1 yields lowest hallucination rate; SileroVAD preprocessing significantly reduces both WER and hallucinations; post-processing text filtering using BoH.
- **Relevance**: Beam_size=1 reducing hallucinations suggests the decoder's search process amplifies errors from bad encoder representations. Our pulse aims to fix encoder representations upstream of these decoder-side fixes.

### 2.3 Listen Like a Teacher (AAAI 2025)
- **Title**: "Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation"
- **Link**: [arXiv](https://arxiv.org/abs/2511.14219)
- **Mechanism**: Two-stage: (1) Adaptive Layer Attention (ALA) groups encoder layers into semantically coherent blocks, fuses with learnable multi-head attention. (2) Knowledge distillation — student (noisy input) aligns with teacher (clean input) on semantic and attention distributions.
- **Results**: Substantial hallucination reduction and more stable decoder cross-attention patterns.
- **Relevance**: ALA is conceptually related to our approach — both modify encoder representations before the decoder. Their finding that encoder layer grouping matters suggests pulse injection might benefit from layer-specific frequencies. The KD framework could complement pulse injection — distill from clean-audio teacher to noisy-audio+pulse student.

### 2.4 Careless Whisper (ACM FAccT 2024)
- **Authors**: Koenecke et al.
- **Link**: [Paper](https://dl.acm.org/doi/10.1145/3630106.3658996)
- **Key Findings**: ~1% of transcriptions contain fully hallucinated phrases; 38% include explicit harms. Hallucinations disproportionately affect speakers with longer non-vocal durations (e.g., aphasia patients).
- **Relevance**: Motivates our work. Correlation between non-vocal duration and hallucination rate is exactly our target phenomenon.

### 2.5 Lost in Transcription (ACL Findings 2025)
- **Link**: [arXiv](https://arxiv.org/abs/2502.12414)
- **Key Findings**: Introduces Hallucination Error Rate (HER) metric. Low WER can mask dangerous hallucinations. Distribution shift (noise, pitch, time stretching) correlates strongly with HER.
- **Relevance**: Provides theoretical framing — hallucinations arise from distribution shift. Pulse injection can be viewed as mapping out-of-distribution silence representations back toward an in-distribution manifold. **Should adopt HER as evaluation metric.**

### 2.6 Whisper Encoder Representations of Silence
- **Source**: [Attanasio analysis](https://gattanasio.cc/post/whisper-encoder/)
- **Key Finding**: Whisper's encoder representation is NOT noise-invariant — it is highly correlated to non-speech sounds. The encoder produces structured (non-zero) representations for silence, which the decoder hallucinates on.
- **Relevance**: Fundamental to our hypothesis. The encoder produces meaningful but misleading features during silence. Pulse injection aims to restructure these into something the decoder interprets as "no speech."

---

## 3. Learned Noise/Perturbation Injection

### 3.1 MuNG: Multimodal Noise Generator (Most Relevant)
- **Title**: "Explore How to Inject Beneficial Noise in MLLMs"
- **Authors**: Zhang et al.
- **Link**: [arXiv](https://arxiv.org/abs/2511.12917) (Nov 2025)
- **Mechanism**: Learns a small noise generator that produces task-adaptive beneficial noise, injected into **both frozen encoder and decoder**. Only ~1% extra parameters. Reformulates MLLM reasoning as variational inference, where noise suppresses irrelevant semantic components.
- **Results**: Outperforms full fine-tuning, LoRA, and DoRA with only 1-2% parameters.
- **Relevance**: **Single most relevant paper.** Proves that learned noise injection into a frozen encoder-decoder model can outperform full fine-tuning. Our pulse is a structured (oscillatory) version of their general noise. Key insight: **inject into the decoder too, not just the encoder.**
- **Frozen decoder**: Directly solves our problem. MuNG injects noise into decoder input while keeping decoder weights frozen.

### 3.2 NoiseBoost: Perturbation for Hallucination Reduction
- **Authors**: Wu et al.
- **Link**: [arXiv](https://arxiv.org/abs/2405.20081), [GitHub](https://github.com/KaiWU5/NoiseBoost)
- **Mechanism**: Injects feature perturbations to visual features, redistributing attention weight between visual and linguistic tokens. Acts as regularizer preventing over-reliance on language priors.
- **Relevance**: Direct parallel. Whisper hallucinates because the decoder over-relies on language model priors when encoder features (from silence) are uninformative. Perturbation can rebalance encoder/decoder attention.

### 3.3 NEFTune: Noisy Embeddings (ICLR 2024)
- **Authors**: Jain et al.
- **Link**: [arXiv](https://arxiv.org/abs/2310.05914)
- **Mechanism**: Adds uniform random noise to input embeddings during fine-tuning. LLaMA-2-7B: 29.79% → 64.69% on AlpacaEval.
- **Relevance**: Establishes embedding-level noise injection as powerful regularizer. Our pulse is a structured version of this. Variant B (random noise control) tests whether sinusoidal structure adds value over random noise.

---

## 4. Positional Encodings and Frequency Design

### 4.1 Mixed RoPE — Learned Frequency Collapse
- **Source**: [ICLR Blog 2025](https://iclr-blogposts.github.io/2025/blog/positional-embedding/)
- **Finding**: When PE frequencies are made learnable, many collapse to near-zero during training.
- **Relevance**: Warning for our learned omega values — need regularization to prevent frequency degeneration. Consider biologically-motivated frequency bands (theta=4-8Hz, alpha=8-13Hz, beta=13-30Hz) or diversity loss.

### 4.2 FINER: Variable-Periodic Activation (CVPR 2024)
- **Link**: [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_FINER_Flexible_Spectral-bias_Tuning_in_Implicit_NEural_Representation_by_Variable-periodic_CVPR_2024_paper.pdf)
- **Mechanism**: Replaces sin(x) with sin((|x|+1)*x), making period position-dependent.
- **Relevance**: Could upgrade our pulse from fixed-frequency sin(omega*t) to variable-frequency sin((|h|+1)*omega*t), where frequency adapts to hidden state magnitude.

### 4.3 SpectFormer (WACV 2025)
- **Link**: [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Patro_SpectFormer_Frequency_and_Attention_is_What_You_Need_in_a_WACV_2025_paper.pdf)
- **Mechanism**: Uses spectral layers in early transformer layers and attention in later layers.
- **Relevance**: Supports pulse injection primarily in early encoder layers (frequency processing) with later layers handling semantic attention.

### 4.4 LieRE: Lie-Group Rotary Encoding (ICLR 2025)
- **Link**: [arXiv](https://arxiv.org/pdf/2410.06205)
- **Mechanism**: Parameterizes rotary PE via matrix exponential of learnable skew-symmetric matrices. More expressive than standard RoPE.
- **Relevance**: Our sin(omega*t + phi(h)) is a special case of Lie-group rotation in 2D. LieRE could inspire richer pulse formulation where oscillation acts in a learned subspace.

---

## 5. Gated and Content-Aware Perturbation

### 5.1 Dynamic Gated Neuron (DGN)
- **Title**: "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks"
- **Link**: [arXiv](https://arxiv.org/html/2509.03281) (Sept 2025)
- **Mechanism**: Input-dependent modulation of membrane conductance — neurons selectively retain relevant information while suppressing noisy inputs.
- **Relevance**: **Directly inspires gated pulse variant.** Instead of `h' = h + alpha * sin(...)`, use `h' = h + g(h) * alpha * sin(...)` where g(h) is a learned gate: high during silence (pulse active), low during speech (original preserved).

### 5.2 AttentionDrop (April 2025)
- **Link**: [arXiv](https://arxiv.org/abs/2504.12088)
- **Mechanism**: Three variants perturbing self-attention distributions: hard masking, Gaussian blur smoothing, KL consistency regularization.
- **Relevance**: Instead of random attention perturbation, use oscillatory signal to modulate attention distributions. "Blurred attention smoothing" is interesting: pulse could smooth over-peaked decoder attention during silence, preventing hallucination.

### 5.3 BioLogicalNeuron: Homeostatic Regulation (2025)
- **Link**: [Nature](https://www.nature.com/articles/s41598-025-09114-8)
- **Mechanism**: Calcium-driven homeostatic regulation with stability monitoring; triggers targeted noise injection to counteract degradation.
- **Relevance**: Homeostatic regulation maps to alpha learning. Alpha should grow only as needed — a stability-monitoring loss term could penalize alpha growth during speech and encourage it during silence.

---

## 6. Silence Detection and VAD

### 6.1 SileroVAD for Whisper Preprocessing
- **Sources**: [WhisperX](https://github.com/m-bain/whisperX), [Whispy](https://arxiv.org/html/2405.03484v1)
- **Mechanism**: SileroVAD detects speech segments; only detected speech is fed to Whisper.
- **Results**: Significant hallucination reduction, no WER degradation.
- **Relevance**: Current "standard fix" — avoids feeding silence to Whisper. Our approach is fundamentally different: teach the model to handle silence correctly. Matters for real-time streaming (VAD latency) and intra-speech pauses.

### 6.2 Inappropriate Pause Detection via ASR
- **Link**: [arXiv](https://arxiv.org/html/2402.18923v1) (2024)
- **Mechanism**: Treats pause detection as ASR — model produces text with explicit pause tags at detected locations.
- **Relevance**: Silence/pause awareness can be integrated into ASR itself. Pulse injection provides an internal "clock" encoding temporal position within silent regions.

---

## 7. Continuous-Time and Neural ODE Approaches

### 7.1 Closed-form Continuous-time Networks (CfC) / Liquid Neural Networks
- **Authors**: Hasani, Lechner et al. (MIT CSAIL)
- **Link**: [Nature MI](https://www.nature.com/articles/s42256-022-00556-7), [GitHub](https://github.com/raminmh/CfC)
- **Mechanism**: Closed-form solution to liquid time-constant neural network dynamics. Explicit time dependence enables 100x faster training than Neural ODEs.
- **Relevance**: Our pulse can be viewed as a special case of CfC where dynamics are sinusoidal. The adaptive time-constant idea (tau per neuron) could inspire time-constant-modulated pulse where tau adapts to speech rate.

### 7.2 LoRA-Whisper (2024)
- **Link**: [arXiv](https://arxiv.org/html/2406.06619v1)
- **Mechanism**: LoRA adapters in Q, K, V projections per Whisper block. Only adapters trained.
- **Results**: 22.97% WER vs 28.33% for full fine-tuning on new languages.
- **Relevance**: Complementary to pulse injection. LoRA modifies weight matrices; pulse modifies hidden states. Could combine: LoRA for language adaptation + pulse for silence robustness.

---

## Synthesis: Novel Architecture Directions

Based on this review, four novel architectures emerge:

### A. Cross-Attention Pulse Injection
Inject pulse at the encoder→decoder cross-attention boundary — where the decoder "reads" encoder representations. Pulse modulates how the decoder interprets encoder output rather than modifying the encoder output itself. Both encoder and decoder remain frozen.

**Supported by**: MuNG (decoder-side injection works), NoiseBoost (perturbation rebalances cross-modal attention).

### B. Gated Content-Aware Pulse
`h' = h + g(h) · alpha · A · sin(omega*t + phi(h))` where g(h) is a learned sigmoid gate that activates during silence and suppresses during speech. Current approach applies pulse uniformly — gating makes it content-dependent.

**Supported by**: DGN (input-dependent gating), BioLogicalNeuron (homeostatic regulation).

### C. Decoder Head Pulse (Targeted)
Instead of encoder injection, inject oscillatory signals specifically into decoder self-attention heads #1, #6, #11 — the hallucination-causing heads identified by Calm-Whisper. Pulse regularizes their attention during silence without modifying weights.

**Supported by**: Calm-Whisper (head localization), AttentionDrop (attention distribution perturbation).

### D. Variational Oscillatory Generator
Replace fixed-form `A*sin(omega*t+phi)` with a small generative network that produces structured oscillatory noise conditioned on the relationship between encoder output and decoder state. Trained end-to-end with frozen backbone via variational inference.

**Supported by**: MuNG (variational noise generator), AKOrN (oscillatory binding).

### Recommended Priority
1. **C (Decoder Head Pulse)** — quickest win, directly targets known hallucination source
2. **A+B (Gated Cross-Attention Pulse)** — most novel, strongest research contribution
3. **D (Variational Generator)** — most ambitious, highest potential payoff
