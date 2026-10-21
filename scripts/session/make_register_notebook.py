"""
Generates .docs/register_settling.ipynb (without outputs).

Run it, then execute the notebook:
    python scripts/session/make_register_notebook.py
    jupyter nbconvert --to notebook --execute --inplace .docs/register_settling.ipynb
The saved notebook's closing cell was then filled in with that run's numbers by hand,
so regenerating it brings back the generic closing text.
"""

from pathlib import Path

import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []
cells.append(md("""# φ-Register: where self-timed settling saves time

Runs on CPU in a few minutes; a GPU is not needed.

Every Fibonacci-encoded weight multiplies by shift-and-add: each 1 digit of the weight's level adds a partial product `x · F(k)` to an accumulator.
This notebook runs that dot product for real conv-layer weights in two accumulators, a φ-register and a binary register (`phi_prune.register.dot`), and asks where the φ side comes out ahead:

1. **Fewer additions.** A Zeckendorf level has fewer 1 digits than the same level in binary, so fewer partial products.
2. **Self-timed settling.** Each φ addition stops when no `11` and no 2 are left; a clocked binary adder waits its worst case every time.
3. **Free integrity.** The same no-`11` scan flags corrupted accumulator states; a binary accumulator cannot.

Settling is counted in rounds, every cell rewriting at once (`normalize_rounds`). Rounds compare the rule systems, not gate-level circuits."""))
cells.append(md("## 0. Install"))
cells.append(code("""# phi_prune.register ships on branch claude/kind-hopper-juyjmk until it is merged into main
import importlib.util, subprocess, sys
if importlib.util.find_spec("phi_prune.register") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/ezexe/phi-prune.git@claude/kind-hopper-juyjmk"])"""))
cells.append(md("""## 1. One pruned, encoded conv layer

ResNet-18's `layer3.0.conv1` (256 output channels, fan-in 128 × 3 × 3 = 1152) with torchvision's ImageNet weights, φ-pruned and encoded per output channel with 10 Fibonacci digits (144 levels), as in the README pipeline.
Without network access to the weights, the cell falls back to Laplace-distributed synthetic weights of the same shape and says so."""))
cells.append(code("""import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from phi_prune import prune
from phi_prune.encoding import FibonacciEncoder
from phi_prune.masks import phi_mask
from phi_prune.register import BINARY, KINDS, PHI, dot, flip_detected, from_int, normalize_rounds, place_values, pour

LAYER = "layer3.0.conv1"
N_DIGITS = 10
SEED = 0
torch.manual_seed(SEED)

try:
    import torchvision
    model = torchvision.models.resnet18(weights="DEFAULT").eval()
    SOURCE = "torchvision ResNet-18 ImageNet weights"
except Exception as exc:  # no torchvision, or no network for the weights
    model = None
    SOURCE = f"synthetic Laplace weights ({type(exc).__name__}: no pretrained weights)"

if model is not None:
    pruned, masks = prune(model)
    weight = dict(pruned.named_parameters())[LAYER + ".weight"].data
    mask = masks[LAYER + ".weight"]
else:
    u = torch.rand(256, 128, 3, 3) - 0.5
    weight = -torch.sign(u) * torch.log1p(-2 * u.abs())
    mask = phi_mask(weight)

kept = [c for c in range(weight.shape[0]) if mask[c].any()]
FAN_IN = weight[0].numel()
print(SOURCE)
print(f"{LAYER}: {weight.shape[0]} output channels, {len(kept)} kept, fan-in {FAN_IN}")"""))
cells.append(md("""## 2. Activations

The layer's real inputs on CIFAR-10 test images at 224 px when the pretrained model and the dataset are available, else on random images (pretrained model) or half-normal values (synthetic).
Each 3 × 3 patch is one activation vector; activations are quantized to 0–255."""))
cells.append(code("""N_PATCHES = 4

def layer_inputs(model, images):
    seen = []
    hook = dict(model.named_modules())[LAYER].register_forward_hook(lambda m, i, o: seen.append(i[0]))
    with torch.no_grad():
        model(images)
    hook.remove()
    conv = dict(model.named_modules())[LAYER]
    patches = F.unfold(seen[0], conv.kernel_size, padding=conv.padding, stride=conv.stride)  # [N, fan-in, L]
    return patches.permute(0, 2, 1).reshape(-1, patches.shape[1])

if model is not None:
    try:
        ds = torchvision.datasets.CIFAR10("data", train=False, download=True)
        imgs = torch.tensor(np.stack([np.asarray(ds[i][0]) for i in range(2)])).permute(0, 3, 1, 2).float() / 255
        mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
        imgs = (F.interpolate(imgs, size=224, mode="bilinear") - mean[:, None, None]) / std[:, None, None]
        ACT_SOURCE = "CIFAR-10 test images"
    except Exception as exc:
        imgs = torch.randn(2, 3, 224, 224)
        ACT_SOURCE = f"random images ({type(exc).__name__}: no CIFAR-10)"
    patches = layer_inputs(model, imgs)
else:
    patches = torch.randn(1024, FAN_IN).abs()
    ACT_SOURCE = "half-normal synthetic activations"

g = torch.Generator().manual_seed(SEED)
pick = patches[torch.randperm(len(patches), generator=g)[:N_PATCHES]]
acts = [np.round((p / p.max() * 255).clamp(0, 255).numpy()).astype(int) for p in pick]
print(ACT_SOURCE)
print(f"{N_PATCHES} patches; nonzero activations per patch: {[int((a > 0).sum()) for a in acts]} of {FAN_IN}")"""))
cells.append(md("""## 3. The dot products, in both registers

For a sample of kept output channels and every patch: the weight levels (`encoder.levels`), then `dot` in a φ-register and in a binary register.
Both get the same integer levels and activations, so both accumulate the same sum; the φ side writes each level in Zeckendorf digits (10 cells), the binary side in 8 bits."""))
cells.append(code("""N_CHANNELS = 8

def run(n_digits):
    enc = FibonacciEncoder(n_digits)
    encoded, scale, _, offset = enc.encode_tensor(weight, mask, return_offset=True, axis=0)
    rows = kept[:: max(1, len(kept) // N_CHANNELS)][:N_CHANNELS]
    out = {kind: {"adds": 0, "rounds": [], "accs": []} for kind in KINDS}
    for c in rows:
        levels = enc.levels(encoded[c], scale[c], offset[c])
        for xs in acts:
            expected = int(np.dot(levels, xs))
            for kind in KINDS:
                acc, rounds = dot(levels, xs, kind)
                assert sum(d * p for d, p in zip(acc, place_values(kind, len(acc)))) == expected
                out[kind]["adds"] += len(rounds)
                out[kind]["rounds"] += rounds
                out[kind]["accs"].append(acc)
    n_dots = len(rows) * len(acts)
    out["n_dots"], out["enc"] = n_dots, enc
    return out

t0 = time.time()
res = run(N_DIGITS)
print(f"{res['n_dots']} dot products in {time.time() - t0:.0f} s")
for kind in KINDS:
    r = res[kind]
    print(f"{kind:>6}: {r['adds'] / res['n_dots']:7.0f} additions per dot product, "
          f"{np.mean(r['rounds']):.2f} rounds per addition on average, {max(r['rounds'])} at worst")"""))
cells.append(md("""## 4. Cost per dot product, four ways to time it

- **φ, self-timed:** the rounds each addition actually took, plus the φ side's extra work, the partial products `x · F(k)`. Each activation needs a ladder of `n_digits − 1` additions (`x·F(k) = x·F(k−1) + x·F(k−2)`), shared by every kept output channel that reads it; the cell runs those ladders and spreads their rounds over the kept channels.
- **Binary, self-timed:** the rounds each binary addition took (an asynchronous adder with carry-completion detection).
- **Binary, clocked ripple:** every addition waits for a carry across the whole accumulator, its width in bits.
- **Binary, clocked lookahead:** every addition waits ⌈log₂ width⌉, the depth of a carry-lookahead adder, the closest of the four to a CPU's adder."""))
cells.append(code("""def ladder_rounds(x, n_digits):
    \"\"\"Rounds to build x·F(0) .. x·F(n_digits-1) in φ-registers by the ladder.\"\"\"
    terms, total = [from_int(x, PHI)], 0
    a, r = normalize_rounds(pour(terms[0], terms[0]), PHI)  # x·2
    terms.append(a); total += r
    for _ in range(n_digits - 2):
        a, r = normalize_rounds(pour(terms[-1], terms[-2]), PHI)
        terms.append(a); total += r
    return total

def cost_table(res, n_digits):
    n = res["n_dots"]
    max_sum = FAN_IN * res["enc"].max_value * 255
    width = max_sum.bit_length()
    ladder = sum(ladder_rounds(int(x), n_digits) for xs in acts for x in xs if x) / len(acts)
    ladder_per_dot = ladder / len(kept)
    rows = {
        "φ, self-timed": sum(res[PHI]["rounds"]) / n + ladder_per_dot,
        "binary, self-timed": sum(res[BINARY]["rounds"]) / n,
        "binary, clocked ripple": res[BINARY]["adds"] / n * width,
        "binary, clocked lookahead": res[BINARY]["adds"] / n * math.ceil(math.log2(width)),
    }
    return rows, width, ladder_per_dot

rows, width, ladder_per_dot = cost_table(res, N_DIGITS)
print(f"binary accumulator: {width} bits; φ ladder adds {ladder_per_dot:.0f} rounds per dot product\\n")
base = rows["φ, self-timed"]
print(f"{'timing':<28}{'rounds / dot product':>22}{'vs φ self-timed':>18}")
for name, v in rows.items():
    print(f"{name:<28}{v:>22,.0f}{v / base:>17.2f}×")"""))
cells.append(md("""## 5. The low end: fewer digits

The same layer at 6, 8 and 10 Fibonacci digits (21, 55 and 144 levels).
Fewer levels mean narrower weights, so fewer additions on both sides, but the φ side's share of 1 digits stays lower than binary's at every width."""))
cells.append(code("""print(f"{'digits':>6}{'levels':>8}{'φ adds':>9}{'bin adds':>10}{'φ self-timed':>14}{'bin self-timed':>16}{'bin lookahead':>15}{'bin ripple':>12}")
for n_digits in (6, 8, 10):
    r = res if n_digits == N_DIGITS else run(n_digits)
    t, _, _ = cost_table(r, n_digits)
    print(f"{n_digits:>6}{r['enc'].n_levels:>8}{r[PHI]['adds'] / r['n_dots']:>9.0f}{r[BINARY]['adds'] / r['n_dots']:>10.0f}"
          f"{t['φ, self-timed']:>14,.0f}{t['binary, self-timed']:>16,.0f}{t['binary, clocked lookahead']:>15,.0f}{t['binary, clocked ripple']:>12,.0f}")"""))
cells.append(md("""## 6. Integrity of the accumulator

Every settled accumulator state from section 3, with each of its bits flipped in turn: the share of flips the no-`11` scan catches.
A binary accumulator reads every pattern as a valid number, so it catches none."""))
cells.append(code("""for kind in KINDS:
    caught = total = 0
    for acc in res[kind]["accs"]:
        for i in range(len(acc)):
            flipped = list(acc)
            flipped[i] ^= 1
            caught += flip_detected(flipped, kind)
            total += 1
    print(f"{kind:>6}: {caught / total:.1%} of {total} single-bit flips detected")"""))
cells.append(md("""## 7. What this run shows

Read the numbers above from your own run; on the run saved in this notebook (see the source lines printed in sections 1 and 2):

- **Fewer additions** is the first saving and does not depend on timing: Zeckendorf levels carry fewer 1 digits than binary ones.
- **Against a clocked binary adder** the self-timed φ accumulator needs fewer rounds per dot product, far fewer than a ripple adder and fewer than a lookahead adder.
- **Against a self-timed binary adder** it does not win: binary additions settle in fewer rounds, which outweighs the φ side's fewer additions.
- **Integrity** comes free on the φ side only.

So the optimization is real where the alternative is a clocked binary MAC, and it comes from two places at once: the low density of 1s in Zeckendorf levels and paying each addition's own settling time. It disappears against binary hardware that is also self-timed, and none of this is a gate-level or silicon measurement."""))
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
nbf.write(nb, Path(__file__).resolve().parents[2] / ".docs" / "register_settling.ipynb")
