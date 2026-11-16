"""
Fibonacci weight encoding.

Quantize neural network weights to values representable as sums of
non-consecutive Fibonacci numbers (Zeckendorf representation). This
enables shift-and-add multiplication (no multiplier circuits) and
provides self-delimiting bitstreams for serialization.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional


# Pre-compute Fibonacci sequence
_FIB_CACHE = [1, 2]
for _i in range(2, 50):
    _FIB_CACHE.append(_FIB_CACHE[-1] + _FIB_CACHE[-2])


class FibonacciEncoder:
    """
    Encode scalar values as Zeckendorf (Fibonacci-base) representations.

    Given n_digits, builds a quantization grid of all integers representable
    as sums of non-consecutive Fibonacci numbers using those digits. Weights
    are mapped to the nearest grid point.

    Args:
        n_digits: Number of Fibonacci digits. More digits → finer quantization.
                  8 digits → 55 levels (default, good for INT8-equivalent precision)
                  10 digits → 144 levels
                  12 digits → 377 levels
    """

    def __init__(self, n_digits: int = 8):
        self.n_digits = n_digits
        self.fibs = _FIB_CACHE[:n_digits]

        # Build the full Zeckendorf grid: all valid sums (no adjacent 1s)
        grid = set()
        self._enumerate(0, 0, False, grid)
        self.grid = np.array(sorted(grid), dtype=np.float64)
        self.n_levels = len(self.grid)
        self.max_value = int(self.grid[-1])  # largest representable value

    def _enumerate(self, idx: int, current_sum: int, prev_used: bool, out: set):
        """Recursively enumerate all valid Zeckendorf sums."""
        if idx >= self.n_digits:
            out.add(current_sum)
            return
        # Skip this digit
        self._enumerate(idx + 1, current_sum, False, out)
        # Use this digit (only if previous wasn't used)
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
        Quantize a weight tensor to the Fibonacci grid.

        Scales weights to [0, max_value], snaps to nearest grid point,
        scales back. Only encodes positions where mask == 1 (if provided).

        Args:
            tensor: Weight tensor to encode
            mask: Optional binary mask (only encode nonzero positions)
            return_offset: Also return the offset, the smallest encoded
                value (0.0 when the mask keeps nothing), which lands on
                level 0. (scale, offset) is the per-layer pair that
                export_bitstream, save_checkpoint and cassini_check take
                in scales
            axis: Give each slice along this axis (0 = output channels)
                its own scale and offset instead of one for the whole
                tensor. One range per tensor spends most of the levels
                on a few outliers when weights are heavy-tailed: encoding
                torchvision's ResNet V2 ImageNet weights that way changes
                the top-1 class of every probed image, per channel keeps
                88-95% of them (scripts/probe_pretrained_encoding.py).
                scale and offset then come back as float64 arrays shaped
                to broadcast against tensor (size 1 except along axis),
                so (w - offset) * scale works as with the scalar pair

        Returns:
            (encoded_tensor, scale_factor, rmse), with the offset as a
            fourth value when return_offset is set
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
        # Snap each value to nearest grid point
        indices = np.searchsorted(self.grid, scaled, side="left")
        indices = np.clip(indices, 0, len(self.grid) - 1)

        # The nearest grid point is grid[indices] or the one below it
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
        """encode_tensor on each slice along axis; (encoded, scale, rmse, offset)."""
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
        Grid levels of an encoded tensor's kept weights, in mask order.

        Inverts encode_tensor's scale-back: round((w - offset) * scale),
        clamped to [0, max_value]. scale and offset are the scalars or
        the broadcastable per-slice arrays encode_tensor returned.

        Returns:
            int64 array, one level per position where mask is set (every
            position when mask is None), in row-major order
        """
        w = tensor.detach().cpu().double().numpy()
        grid = np.round((w - np.asarray(offset, dtype=np.float64)) * np.asarray(scale, dtype=np.float64))
        grid = np.broadcast_to(grid, w.shape)
        if mask is not None:
            grid = grid[mask.detach().cpu().bool().numpy()]
        return np.clip(grid, 0, self.max_value).astype(np.int64).ravel()

    def to_codeword(self, grid_value: int) -> list:
        """
        Convert a grid integer to its Zeckendorf digit list.

        Args:
            grid_value: Integer that is a valid Fibonacci sum

        Returns:
            List of 0s and 1s (LSB first), length = n_digits
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
        Serialize codewords into a self-delimiting bitstream.

        Standard Fibonacci coding: each codeword is written from the
        most significant set bit down to LSB, then a '1' is appended.
        Since valid Zeckendorf representations never contain '11', the
        trailing '1' after the MSB (always 1) creates a unique '11'
        delimiter.

        Values of 0 are handled by encoding grid_value + 1 internally,
        so the minimum encoded value is 1 (which always has a set MSB).

        Args:
            codewords: List of Zeckendorf digit lists (from to_codeword)

        Returns:
            String of '0's and '1's, parseable without length headers
        """
        bits = []
        for cw in codewords:
            # Find the MSB (highest set bit)
            msb = len(cw) - 1
            while msb > 0 and cw[msb] == 0:
                msb -= 1
            # Write LSB to MSB (standard Fibonacci coding order),
            # then append '1' as delimiter
            for i in range(msb + 1):
                bits.append(cw[i])
            bits.append(1)  # delimiter — creates '11' with MSB
        return "".join(str(b) for b in bits)

    def encode_to_stream(self, grid_values: list) -> str:
        """
        Encode a list of grid values to a self-delimiting bitstream.

        Adds +1 offset so that 0 is encodable (becomes 1, which has a set MSB).
        Uses n_digits+1 internally to accommodate the offset at the top end.
        Use decode_from_stream to reverse.
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
        Decode a self-delimiting bitstream back to grid values.

        Reverses the +1 offset applied by encode_to_stream.
        """
        codewords = self.parse_bitstream(bitstream, self.n_digits + 1)
        fibs_ext = _FIB_CACHE[:self.n_digits + 1]
        values = []
        for cw in codewords:
            val = sum(f * d for f, d in zip(fibs_ext, cw))
            values.append(val - 1)  # undo +1 offset
        return values

    @staticmethod
    def parse_bitstream(bitstream: str, n_digits: int) -> list:
        """
        Parse a self-delimiting Fibonacci bitstream back into codewords.

        Scans for '11' delimiter pattern to find codeword boundaries.

        Args:
            bitstream: String of '0's and '1's
            n_digits: Expected codeword width

        Returns:
            List of Zeckendorf digit lists (LSB-first, padded to n_digits)
        """
        codewords = []
        current = []
        prev = "0"
        for bit in bitstream:
            if bit == "1" and prev == "1":
                # Delimiter found — current (without the last '1' which
                # is the MSB, not the delimiter) is the codeword body.
                # The '1' already in current[] is the MSB — keep it.
                cw = list(current)  # already LSB-first
                # Pad to n_digits
                cw = (cw + [0] * n_digits)[:n_digits]
                codewords.append(cw)
                current = []
                prev = "0"
            else:
                current.append(int(bit))
                prev = bit
        return codewords
