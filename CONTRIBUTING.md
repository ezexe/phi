**A. Edge deployment — shrink a model for devices without NVIDIA GPUs**

You have a trained model that's too large for a Raspberry Pi, Jetson Nano, or mobile phone. Prune it to 50% density, fine-tune, deploy. The inference code is standard PyTorch — zeros are zeros, no special runtime. Model file is half the effective size. Works on ARM, x86, AMD, anything.

**B. FPGA inference — eliminate multipliers**

After pruning, Fibonacci-encode the surviving weights. Each weight becomes a sum of non-consecutive Fibonacci numbers. Multiplication by that weight becomes shift-and-add (shift along the Fibonacci sequence, add the results). On an FPGA, this replaces multiplier blocks (DSP slices) with adder trees — 86% area reduction per the published literature. You fit a larger model on the same FPGA, or a same-size model on a cheaper FPGA.

**C. Safety-critical inference — free corruption detection**

Automotive, medical, aerospace. Radiation causes single-event upsets that flip bits in weight SRAM. Load a Fibonacci-encoded checkpoint, and on every weight read, scan for adjacent 1s. If found, the weight is corrupted. No CRC, no ECC overhead — just one bitwise check. Catches 46–48% of single-bit flips for free on pretrained ResNets (32% with the sign-magnitude codes of the original experiment). Layer lightweight ECC on top for the rest and your total protection cost drops.

**D. Model distribution — smaller downloads, self-framing streams**

Publish your pruned model as a Fibonacci bitstream. The format is self-delimiting (each codeword boundary is found by scanning for "11"), so the decoder doesn't need length headers or a schema. Corrupt bytes in transit don't propagate — you lose one weight, not the rest of the tensor. The mask compresses to 0.694 bits/position (vs 1.0 for arbitrary 50% masks) because the RLL structure has a known optimal encoding. Smaller OTA updates for edge fleets.

**E. Federated learning — resilient weight updates**

Clients send pruned weight deltas to the server. Self-delimiting format means partial uploads are still parseable — if the connection drops, everything received so far decodes cleanly. Adjacency check on arrival catches transmission errors without a full checksum pass.

**F. Neural architecture search — analytic capacity budgets**

The φ mask has a known information capacity: log₂(φ) ≈ 0.694 bits per position. Before training, you can calculate the exact representational budget of each layer. Allocate capacity across layers mathematically rather than by trial-and-error pruning ratios. No other pruning method gives you a closed-form capacity bound.

**G. Adversarial robustness — spectral defense**

Phase 2 showed φ-pruning pushes 87% of activation energy into low frequencies (vs 52% dense). Phase 4 is testing whether this translates to adversarial resistance. If it does: prune for compression, get robustness for free. No adversarial training (2× compute cost), no input preprocessing, no ensembles.

**H. Knowledge distillation — smoother teacher**

A φ-pruned teacher produces smoother soft labels (less high-frequency noise in the output distribution). Smoother soft labels are easier for a student network to learn from. A pruned teacher might be a better distillation source than the dense original despite lower accuracy.

**I. Quantization-aware training — Fibonacci as the quantization grid**

Quantize to Fibonacci levels (55 levels at 8 digits, 144 at 10 digits). The levels themselves are uniform — every integer from 0 to F(n+2) − 1 has a Zeckendorf form — so accuracy matches uniform quantization with the same number of levels; what changes is the digits, which need fewer 1s per weight than binary (see `scripts/transformer_check.py`). Combine with pruning: prune first (removes 50% of weights), then quantize survivors to Fibonacci grid. Both operations respect the same constraint.

**J. Model integrity verification — deployment pipeline checksums**

Before deploying a model update to production, run `phi-prune check model.pt`. One command verifies every weight in every pruned layer hasn't been corrupted since training. Integrates into CI/CD — fail the deployment if any mask is violated. No cryptographic hashing needed for structural integrity (use hashing separately for authentication).