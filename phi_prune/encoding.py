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
        if span < 1e-10:
            return out(tensor.clone(), 1.0, 0.0, vmin)

        scale = self.max_value / span
        offset = vmin

        # Quantize
        active_np = active.cpu().numpy()
        scaled = (active_np - offset) * scale
        # Round each value to the nearest quantization level
        indices = np.searchsorted(self.grid, scaled, side="left")
        indices = np.clip(indices, 0, len(self.grid) - 1)

        # The nearest level is grid[indices] or the one below it
        upper = self.grid[indices]
        lower = self.grid[np.maximum(indices - 1, 0)]
        use_lower = np.abs(scaled - lower) < np.abs(scaled - upper)
        quantized = np.where(use_lower, lower, upper)

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

    def to_bitstream(self, codewords: list) -> str:
        """
        Write digit lists as a Fibonacci code: each from its least significant digit up to its highest
        set digit, followed by a 1.

        A Zeckendorf representation never holds 11, so the appended 1 and the highest set digit make
        the only 11 of each codeword, its comma, and a reader splits the stream there with no lengths.
        A digit list with no set digit, quantization level 0, comes out as 01 with no 11 of its
        own and runs into the codeword after it, which is why encode_to_stream writes each level
        plus 1.

        Args:
            codewords: Zeckendorf digit lists, as to_codeword returns them

        Returns:
            String of '0's and '1's
        """
        bits = []
        for cw in codewords:
            # The highest set digit
            msb = len(cw) - 1
            while msb > 0 and cw[msb] == 0:
                msb -= 1
            # Least significant digit first, as Fibonacci coding writes it,
            # then the 1 that ends the codeword
            for i in range(msb + 1):
                bits.append(cw[i])
            bits.append(1)  # the appended 1, which with the highest set digit makes the comma, 11
        return "".join(str(b) for b in bits)

    def encode_to_stream(self, grid_values: list) -> str:
        """
        Write quantization levels as a Fibonacci code of each level plus 1, so that level 0 has a codeword too.

        max_value + 1 needs one more digit, so the codewords run up to n_digits + 1 digits.
        decode_from_stream reads the stream back.
        """
        # Need one extra digit to handle max_value + 1
        fibs_ext = _FIB_CACHE[:self.n_digits + 1]
        codewords = []
        for v in grid_values:
            val = v + 1
            digits = [0] * (self.n_digits + 1)
            rem = val
            for i in range(self.n_digits, -1, -1):
                if fibs_ext[i] <= rem:
                    digits[i] = 1
                    rem -= fibs_ext[i]
            codewords.append(digits)
        return self.to_bitstream(codewords)

    def decode_from_stream(self, bitstream: str) -> list:
        """
        Read a stream that encode_to_stream wrote back to quantization levels, taking off the 1 it added.
        """
        codewords = self.parse_bitstream(bitstream, self.n_digits + 1)
        fibs_ext = _FIB_CACHE[:self.n_digits + 1]
        values = []
        for cw in codewords:
            val = sum(f * d for f, d in zip(fibs_ext, cw))
            values.append(val - 1)  # take off the 1 encode_to_stream added
        return values

    @staticmethod
    def parse_bitstream(bitstream: str, n_digits: int) -> list:
        """
        Split a Fibonacci-coded stream into digit lists at each 11, the comma that ends a codeword.

        Args:
            bitstream: String of '0's and '1's
            n_digits: Width each digit list is padded or cut to

        Returns:
            Zeckendorf digit lists, least significant digit first, n_digits long
        """
        codewords = []
        current = []
        prev = "0"
        for bit in bitstream:
            if bit == "1" and prev == "1":
                # This 1 and the 1 before it make the comma, 11, which ends the codeword;
                # current holds the codeword's digits, the highest set digit last
                cw = list(current)  # least significant digit first
                # Pad or cut to n_digits
                cw = (cw + [0] * n_digits)[:n_digits]
                codewords.append(cw)
                current = []
                prev = "0"
            else:
                current.append(int(bit))
                prev = bit
        return codewords
