# phi-prune

φ-pruning: structured sparsity from the φ-register's adjacency constraint, **no two adjacent weights active**.

One rule from one matrix (`M = [[1,1],[1,0]]`) gives you:
- **Pruning** — About 50% structured sparsity; on ResNet-20/CIFAR-10 it lands 0.50 points behind a mask that keeps 2 of every 4 channels (a channel-level pattern, not NVIDIA's weight-level 2:4)
- **Encoding** — Weights quantized to Fibonacci-coded levels, meant for shift-and-add multiplication (no such kernel yet; the 86% multiplier-area saving below is an estimate, not a measurement)
- **Integrity** — Free corruption detection via adjacency check (32% of single-bit flips caught with the original experiment's 8-digit sign-magnitude codewords on ResNet-20, 46–48% with this package's 10-digit codewords on eight larger ResNets; no parity bits)
- **Serialization** — Self-delimiting bitstreams with no length headers (on eight ResNets the stream took 9.7–10.3 bits per weight, more than the 7.2 bits of a fixed-width code for the same 144 levels)

No NVIDIA hardware required. No sparse tensor cores. The constraint is simple enough for any architecture to exploit.

## Quick Start

```python
from phi_prune import prune, finetune, check

# 1. Prune
model, masks = prune(trained_model)
# → 50% density, adjacency-constrained

# 2. Fine-tune (mask-aware: gradients zeroed on pruned positions; amp=True runs fp16 on CUDA)
finetune(model, train_loader, epochs=40, masks=masks, val_loader=test_loader)

# 3. Verify
report = check(model, masks)
assert report["_summary"]["all_masks_valid"]
```

## Fibonacci Encoding

```python
from phi_prune import FibonacciEncoder

encoder = FibonacciEncoder(n_digits=8)  # 55 quantization levels
# Encode surviving weights, layer by layer
scales = {}  # name → (scale, offset) for cassini_check, save_checkpoint, export_bitstream
for name, param in model.named_parameters():
    if name in masks:
        # offset: the smallest kept weight, which lands on level 0
        # axis=0: a scale and offset per output channel (arrays); one pair per layer if omitted
        encoded, scale, rmse, offset = encoder.encode_tensor(param.data, mask=masks[name],
                                                             return_offset=True, axis=0)
        param.data.copy_(encoded)
        scales[name] = (scale, offset)
# → Weights are now sums of non-consecutive Fibonacci numbers
# → Multiplication = shift-and-add (no multiplier needed)

from phi_prune import evaluate
acc = evaluate(model, test_loader, amp=True)  # fp16 on CUDA; batches whose logits overflow re-run in fp32
```

## Integrity Checking

```python
from phi_prune import cassini_check

# Check all weights for corruption (free — just scan for adjacent 1s)
report = cassini_check(model, masks, encoder, scales)
print(f"Pass rate: {report['_aggregate']['pass_rate']:.1%}")
```

## Export

```python
from phi_prune.export import save_checkpoint, export_onnx, export_bitstream, load_bitstream

# PyTorch checkpoint (includes masks + metadata)
save_checkpoint(model, masks, "model_phi.pt", encoder=encoder, scales=scales)

# ONNX (sparse weights baked in, runs on any runtime)
export_onnx(model, "model_phi.onnx")

# Self-delimiting bitstream (minimal format for edge deployment)
stats = export_bitstream(model, masks, encoder, scales, "model.phi")
print(f"{stats['bits_per_weight']:.1f} bits/weight")
# Read it back: decodes the weights into the masked positions of a model with the same masks
load_bitstream(model, masks, "model.phi")
```

## Sparse Deltas

For a model trained in rounds, ship each round's update instead of the whole model: the largest fraction of each tensor's change, positions as Fibonacci gap codes (which beat Elias-gamma, Elias-delta and fixed-width positions on fine-tuning updates, `.docs/register.md` section 5d), values as a sign and a Fibonacci-coded magnitude level.

```python
from phi_prune.delta import encode_delta, apply_delta, save_delta, load_delta

header, payload = encode_delta(base.state_dict(), tuned.state_dict(), keep=0.01, n_digits=6)
save_delta("round-01.phid", header, payload)

state = base.state_dict()
apply_delta(state, *load_delta("round-01.phid"))  # checks counts, positions and shapes before writing
base.load_state_dict(state)
```

The codes alone are in `phi_prune.gaps` (`encode_positions`, `decode_positions`, `gap_bits`). [`.docs/p4/continual_p4.ipynb`](.docs/p4/continual_p4.ipynb) sketches the rounds on a `.claude/p4` store.

## CLI

```bash
# Verify checkpoint integrity
phi-prune check model_phi.pt

# Print pruning statistics
phi-prune info model_phi.pt
```

## Benchmark Results (ResNet-20, CIFAR-10)

| Method | Accuracy | Drop | Density |
|--------|----------|------|---------|
| Dense baseline | 91.66% | — | 100% |
| φ-pruned | 88.05% | 3.61% | 50.8% |
| 2 of every 4 channels pruned | 88.55% | 3.11% | 50.9% |
| φ-pruned + Fibonacci encoded | 87.45% | 4.21% | 50.8% |

Estimated hardware savings (vs dense binary), not measured: 86% multiplier area and 71% power-delay product, which combine the 50.8% density with the 73% multiplier-area and 43% power-delay-product reductions that DATE 2021 reported for its Fibonacci weight encoding (1 − 0.508 × 0.27 and 1 − 0.508 × 0.57, in `compute_hardware_cost` of [`.docs/experiment/phase1_stacked.py`](.docs/experiment/phase1_stacked.py)).
That paper's encoding keeps weights whose binary form has no two adjacent 1s, while this package stores Zeckendorf digits, for which no circuit cost has been estimated.
Free integrity: 32% single-bit-flip detection via adjacency check, with the experiment's 8-digit sign-magnitude codewords (`abs(round(w / scale))`), which put most weights on the lowest levels. The package encodes `(w - min) * scale` instead and catches more; see below.

## Benchmark Results (ResNets, CIFAR-10 at 224 px, Colab T4)

In short:

- Pruning zeroes about half of each network's conv weights.
  - Why: the adjacency rule never keeps two neighboring output channels of a layer, so at most half of them survive; the pruner keeps the heaviest set the rule allows, which lands just under half (48.8–49.8% here).
- With the original ImageNet weights (V1), 5 epochs of retraining bring every pruned model back to within 2–7 points of its unpruned accuracy.
  - Why: right after pruning every model scores 10%, a single predicted class, yet V1 models climb back to 82–93% in the first retraining epoch alone, so the surviving weights still carry most of what was learned; retraining with the pruned weights held at zero teaches them to cover for the missing channels.
- Among the V1 models, the bigger the network, the smaller the loss: ResNet-18 goes from 94.81% unpruned to 88.05% pruned (6.8 points), ResNet-152 from 97.47% to 95.28% (2.2 points).
  - Why (likely, not tested here): a bigger network has more channels doing overlapping work, so more is left to cover for the ones pruned away; the trend holds even among the three models retrained at the same batch size (ResNet-18, -34 and -50 lose 6.8, 5.2 and 4.2 points).
- With torchvision's newer V2 weights, the same retraining leaves the pruned models 15–29 points below their unpruned accuracy: ResNet-101 goes from 97.53% to 68.35%, and ResNet-152, the best of them, from 97.70% to 82.27%.
  - Why: not established. The pruned V2 models start from the same 10% but climb far more slowly, reaching 47.8%, 39.5% and 58.9% (ResNet-50, -101, -152) after the first retraining epoch, against 85.8%, 91.4% and 92.8% for V1. It is not the φ pattern itself: a mask that keeps 2 of every 4 channels leaves ResNet-50 V2 far behind too (see below). Two untested explanations: pruning removes more of what the V2 weights rely on, or 5 epochs at this small learning rate are too few for them to recover; a longer or higher-learning-rate retraining would tell them apart.

| Model | ImageNet weights | Params | Dense | Pruned | Drop (pts) | Conv density | Pruned convs | Minutes |
|-------|------------------|--------|-------|--------|------------|--------------|--------------|---------|
| ResNet-18 | V1 | 11.2M | 94.81% | 88.05% | 6.76 | 49.8% | 20 | — |
| ResNet-34 | V1 | 21.3M | 96.22% | 91.02% | 5.20 | 49.7% | 36 | 11.2 |
| ResNet-50 | V1 | 23.5M | 96.02% | 91.80% | 4.22 | 49.5% | 53 | 19.8 |
| ResNet-50 | V2 | 23.5M | 96.70% | 70.49% | 26.21 | 49.5% | 53 | 19.8 |
| ResNet-101 | V1 | 42.5M | 97.18% | 94.20% | 2.98 | 48.9% | 104 | 31.7 |
| ResNet-101 | V2 | 42.5M | 97.53% | 68.35% | 29.18 | 49.3% | 104 | 31.2 |
| ResNet-152 | V1 | 58.2M | 97.47% | 95.28% | 2.19 | 48.8% | 155 | 44.6 |
| ResNet-152 | V2 | 58.2M | 97.70% | 82.27% | 15.43 | 49.2% | 155 | 44.1 |

One run of [`.docs/quickstart_gpu.ipynb`](.docs/quickstart_gpu.ipynb) with version 0.2.1 of the package on 2026-09-23: ResNet-18 from its walkthrough sections, the others from its variant cells.
Each model starts from torchvision's ImageNet weights with a new 10-class `fc` head, gets 2 dense fine-tuning epochs, has every conv layer pruned while `fc` stays dense, and gets 5 mask-aware fine-tuning epochs; training is SGD with a cosine schedule under fp16 autocast.
The learning rate is 0.01 for the dense epochs and 0.001 after pruning at batch 128 (ResNet-18 to ResNet-50), halved along with the batch for ResNet-101 and ResNet-152 at batch 64.
Dense and Pruned are the best CIFAR-10 test accuracy over each stage's epochs; before fine-tuning, every pruned model scored 10.00%, a single predicted class.
Minutes is the wall time from the dense fine-tune through the final mask check; ResNet-18 took about a minute per epoch.

### Baseline and encoding

A mask that keeps 2 of every 4 neighboring output channels, the baseline of the ResNet-20 experiment, gives the same ResNet-50 V2 a smaller but still large gap: 96.73% unpruned to 78.70% pruned (18.03 points at 50.0% density), against 26.21 points with the φ mask, one run each.
So the V2 gap is not specific to the φ pattern.
Encoding every pruned layer with 10-digit Fibonacci codewords (144 levels per layer, nearest rounding) costs well under a point on the V1 models and several points on the V2 ones:

| Model | ImageNet weights | Pruned | Encoded | Cost (pts) |
|-------|------------------|--------|---------|------------|
| ResNet-34 | V1 | 91.03% | 90.58% | 0.45 |
| ResNet-50 | V1 | 91.80% | 91.25% | 0.55 |
| ResNet-101 | V1 | 94.21% | 93.93% | 0.28 |
| ResNet-152 | V1 | 95.28% | 94.96% | 0.32 |
| ResNet-50 | V2 | 70.48% | 64.94% | 5.54 |
| ResNet-50 | V2, 2 of every 4 channels | 78.70% | 71.91% | 6.79 |
| ResNet-101 | V2 | 68.36% | 63.90% | 4.46 |
| ResNet-152 | V2 | 82.23% | 10.00% | 72.23 |

What these numbers mean:

- Encoding stores each surviving weight as one of 144 Fibonacci-coded levels instead of a full 32-bit number; on the V1 models that costs well under a point of accuracy, on the V2 ones 4–7 points.
- After encoding, ResNet-152 V2 scores 10.00%, exactly what answering the same class for every image scores on CIFAR-10's ten equally common classes; the likely cause is the encoding's single range per layer. On torchvision's ImageNet weights (`scripts/probe_pretrained_encoding.py`, 64 CIFAR-10 images at 224 px, fp32), encoding every conv layer with one scale and offset per layer changes the top-1 class of every image for ResNet-50, -101 and -152 V2, against 9.4% of images for ResNet-152 V1; a scale and offset per output channel (`encode_tensor(..., axis=0)`, phi-prune 0.2.3) keeps 87.5-95.3% of the V2 predictions, and clipping outlying weights instead does not. No block output came near fp16's limit, so fp16 overflow is not the cause there. A reduced run of the notebook's pipeline on a CPU runner (`scripts/finetune_encode_check.py`: ResNet-152 V2, 5000 CIFAR-10 training images at 128 px, one dense and six mask-aware fine-tune epochs, 2000 test images) reproduces the collapse and the fix, in fp32 and fp16 alike with no non-finite logits: dense 88.75%, pruned 31.70%, encoded with one range per layer 9.25% (chance), encoded per output channel 32.85%. The table above uses one range per layer and awaits a re-run with the notebook's new per-channel default.
- Pruned here is measured again from each saved file rather than copied from training, so it can differ from the main table by up to 0.04 points.
- Flips caught: when one random stored bit is flipped, the flip lands next to another 1 about half the time (46–48% here), and the adjacency check spots it with no extra storage; the other half go unnoticed.
- Stream size: the Fibonacci stream marks where each weight ends, so it needs no length fields, but at 9.7–10.3 bits per weight it is 35–45% larger than a plain fixed-size code for the same 144 levels (7.17 bits); it buys that marking and the flip check, not smaller files.
- Source: the notebook at commit 6f9ef3a, run on a Colab T4 with the same training setup as the main table.

## φ-Register vs Binary Register

`phi_prune.register` models the φ-register and a binary register as local rewrite rules and compares them on phi-prune's own multiply-accumulate step; [`.docs/register.md`](.docs/register.md) walks through the module, its tests, the CPU notebook [`.docs/register_settling.ipynb`](.docs/register_settling.ipynb) and the transformer checks in [`scripts/transformer_check.py`](scripts/transformer_check.py), with a table linking every result to the script that produced it ([`scripts/session/`](scripts/session/)) and its saved output ([`.docs/results/`](.docs/results/)).
In short, on synthetic weights so far: a self-timed φ accumulator needs 15% fewer additions and fewer settling rounds per dot product than a clocked binary one (1.7× against carry-lookahead), catches 41-45% of single-bit flips, and loses to a binary accumulator that is also self-timed (0.82×).

## How It Works

The adjacency constraint ("no two consecutive 1s") is applied along the output channel axis of each layer.
The model's last Conv2d or Linear layer — the classifier head in standard architectures — stays dense by default, because along that axis its positions are the classes and the constraint would keep at most half of them (`prune_head=True` prunes it too; `exclude=` keeps any other layer whole).
A dynamic programming algorithm finds the maximum-weight independent set — the optimal subset of channels to keep given their magnitudes — subject to this constraint.

The same constraint governs the Fibonacci number system: every positive integer has a unique representation as a sum of non-consecutive Fibonacci numbers. Encoding weights in this system enables multiplication via shift-and-add (each Fibonacci number is the sum of two predecessors), and corrupted codewords are detectable by scanning for the forbidden "11" pattern.

All three properties — structured sparsity, efficient arithmetic, free error detection — derive from the single matrix `M = [[1,1],[1,0]]`, the object the φ-register framework is built on.

## Background

The adjacency constraint is the rule of Zeckendorf's theorem (every positive integer is a unique sum of non-consecutive Fibonacci numbers) and of Fibonacci coding, which this package borrows for its weight encoding, integrity check and self-delimiting streams. Earlier versions were published as `zeckendorf-prune`; from 0.3.0 the package is `phi_prune`. Checkpoints saved by earlier versions still load and check.

## Citation

```
@software{phi_prune,
  title={phi-prune: Structured Sparsity via the Adjacency Constraint},
  year={2026},
  url={https://github.com/ezexe/phi}
}
```
