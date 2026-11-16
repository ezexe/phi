"""
Uniform affine quantization of weights to F(n+2) integer quantization levels, each written as its
Zeckendorf representation.

encode_tensor rounds each kept weight to the nearest of the quantization levels 0 to F(n+2) - 1,
spread evenly between the smallest and largest kept weight, per tensor or per channel, and returns
the de-quantized weights, which is fake quantization. A level's Zeckendorf digits hold no two
adjacent 1s, so the Fibonacci code of each level plus 1, its digits followed by a 1, holds a single
11 at its end, and a reader splits a stream of such codewords at each 11 with no lengths written.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional


# Place values of the Zeckendorf digits: 1, 2, 3, 5, 8, ...
_FIB_CACHE = [1, 2]
for _i in range(2, 50):
    _FIB_CACHE.append(_FIB_CACHE[-1] + _FIB_CACHE[-2])


class FibonacciEncoder:
    """
    Quantize weights to integer quantization levels, and write each level as its Zeckendorf digits.

    The levels are every sum of non-adjacent place values among the first n_digits, which is every
    integer from 0 to F(n_digits + 2) - 1, and a weight goes to the nearest one.

    Args:
        n_digits: Number of Zeckendorf digits; more digits give finer levels.
                  8 digits give 55 levels, the default, about 5.8 bits
                  10 digits give 144 levels
                  12 digits give 377 levels
    """

    def __init__(self, n_digits: int = 8):
        self.n_digits = n_digits
        self.fibs = _FIB_CACHE[:n_digits]

        # The quantization levels: every sum of non-adjacent place values, every integer up to F(n + 2) - 1
        grid = set()
        self._enumerate(0, 0, False, grid)
        self.grid = np.array(sorted(grid), dtype=np.float64)
        self.n_levels = len(self.grid)
        self.max_value = int(self.grid[-1])  # the highest quantization level

    def _enumerate(self, idx: int, current_sum: int, prev_used: bool, out: set):
        """Add to out every sum of non-adjacent place values from idx on, recursively."""
        if idx >= self.n_digits:
            out.add(current_sum)
            return
        # Skip this digit
        self._enumerate(idx + 1, current_sum, False, out)
        # Use this digit, when the previous one is unused
        if not prev_used:
            self._enumerate(idx + 1, current_sum + self.fibs[idx], True, out)

    def encode_tensor(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor = None,
        *,
        return_offset: bool = False,
        axis: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, float, float], Tuple[torch.Tensor, float, float, float]]:
        """
        Quantize a weight tensor to the quantization levels.

        Scales the kept weights to [0, max_value], rounds each to the nearest level and scales back.
        Only positions where mask is set are quantized, every position when mask is None.
        Kept weights that span less than 1e-10 all go to level 0, the smallest of them, with a scale of 1.0.

        Args:
            tensor: Weight tensor to quantize
            mask: Optional binary mask; only its kept positions are quantized
            return_offset: Also return the quantization offset, the smallest kept weight, which
                lands on level 0, or 0.0 when the mask keeps nothing. The scale and offset are the
                pair that export_bitstream, save_checkpoint and cassini_check take in scales
            axis: Give each slice along this axis, 0 for output channels, its own scale and offset,
                which is per-channel quantization, instead of one pair for the whole tensor, which
                is per-tensor quantization. Per tensor, a few outlying weights spread the levels
                thin: on torchvision's ResNet-50, 101 and 152 V2 ImageNet weights, quantizing every
                convolution per tensor changed the top-1 class of all 64 probed images, while per
                channel kept 87.5% to 95.3% of them, as
                experiments/phi-prune/results/probe-encoding-36066245824.txt in ezexe/p4-phi holds.
                scale and offset then come back as float64 arrays shaped to broadcast against
                tensor, of size 1 except along axis, so (w - offset) * scale works as with scalars

        Returns:
            encoded_tensor, scale and rmse, with the offset as a fourth value when return_offset
            is set. scale is levels per unit of weight, the inverse of the quantization step
        """
        def out(encoded, scale, rmse, offset):
            return (encoded, scale, rmse, offset) if return_offset else (encoded, scale, rmse)

        if axis is not None:
            return out(*self._encode_per_slice(tensor, mask, axis))

        if mask is not None:
            active = tensor[mask.bool()]
        else:
            active = tensor.flatten()

        if active.numel() == 0:
            return out(tensor.clone(), 1.0, 0.0, 0.0)

        # Scale to [0, max_value]
        vmin, vmax = active.min().item(), active.max().item()
        span = vmax - vmin
        offset = vmin
        active_np = active.cpu().numpy()
        if span < 1e-10:
            # A range this narrow holds one level, so every kept weight goes to level 0, the smallest of them
            scale = 1.0
            quantized = np.zeros(active_np.shape)
        else:
            scale = self.max_value / span
            # The levels are every integer from 0 to max_value, so the nearest is a rounding, ties going up.
            # floor plus a compare of the remainder is exact in every dtype, where scaled + 0.5 can round up
            # past the half. A NaN, which an infinite weight gives, goes to max_value as the grid search sent it
            scaled = (active_np - offset) * scale
            low = np.floor(scaled)
            quantized = np.clip(low + (scaled - low >= 0.5), 0, self.max_value).astype(np.float64)
            quantized[np.isnan(scaled)] = self.max_value

        # Scale back
        decoded = (quantized / scale) + offset
        rmse = float(np.sqrt(np.mean((active_np - decoded) ** 2)))

        # Write back
        result = tensor.clone()
        if mask is not None:
            result[mask.bool()] = torch.tensor(decoded, dtype=tensor.dtype, device=tensor.device)
        else:
            result = torch.tensor(decoded, dtype=tensor.dtype, device=tensor.device).reshape(tensor.shape)

        return out(result, scale, rmse, offset)

    def _encode_per_slice(self, tensor, mask, axis):
        """encode_tensor on each slice along axis, returning encoded, scale, rmse and offset."""
        axis = axis % tensor.dim()
        n = tensor.shape[axis]
        shape = [1] * tensor.dim()
        shape[axis] = n
        scale = np.ones(n, dtype=np.float64)
        offset = np.zeros(n, dtype=np.float64)
        result = tensor.clone()
        sq_err, count = 0.0, 0
        for i in range(n):
            sl = tensor.select(axis, i)
            m = None if mask is None else mask.select(axis, i)
            encoded, scale[i], rmse, offset[i] = self.encode_tensor(sl, m, return_offset=True)
            result.select(axis, i).copy_(encoded)
            k = sl.numel() if m is None else int(m.bool().sum())
            sq_err += rmse ** 2 * k
            count += k
        rmse = float(np.sqrt(sq_err / count)) if count else 0.0
        return result, scale.reshape(shape), rmse, offset.reshape(shape)

    def levels(self, tensor: torch.Tensor, scale, offset, mask: torch.Tensor = None) -> np.ndarray:
        """
        The quantization levels of a quantized tensor's kept weights, in mask order.

        Inverts encode_tensor's scale-back: round((w - offset) * scale), clipped to [0, max_value].
        scale and offset are the scalars or the broadcastable per-slice arrays encode_tensor returned.

        Returns:
            int64 array, one level per position where mask is set, every position when mask is
            None, in row-major order
        """
        w = tensor.detach().cpu().double().numpy()
        grid = np.round((w - np.asarray(offset, dtype=np.float64)) * np.asarray(scale, dtype=np.float64))
        grid = np.broadcast_to(grid, w.shape)
        if mask is not None:
            grid = grid[mask.detach().cpu().bool().numpy()]
        return np.clip(grid, 0, self.max_value).astype(np.int64).ravel()

    def to_codeword(self, grid_value: int) -> list:
        """
        A quantization level's Zeckendorf representation, as a list of digits.

        Args:
            grid_value: A level, an integer from 0 to max_value

        Returns:
            List of 0s and 1s, least significant digit first, length = n_digits
        """
        digits = [0] * self.n_digits
        rem = grid_value
        for i in range(self.n_digits - 1, -1, -1):
            if self.fibs[i] <= rem:
                digits[i] = 1
                rem -= self.fibs[i]
        return digits
