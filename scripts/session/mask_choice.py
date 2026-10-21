"""
How much choice each mask pattern leaves (.docs/register.md, section 0).

Counts the masks a pattern allows on a 64-channel layer, as bits of
choice per channel: C(n - k + 1, k) masks keep k of n channels with no
two adjacent; 2:4 and 1:4 allow 6 and 4 choices per block of 4.
"""

from math import comb, log2

N = 64


def main():
    print(f"{'mask':<28}{'density':>8}{'bits/channel':>14}")
    for density in (0.25, 0.281, 0.375, 0.5):
        k = round(N * density)
        print(f"{'phi':<28}{k / N:>8.3f}{log2(comb(N - k + 1, k)) / N:>14.3f}")
    print(f"{'2:4':<28}{0.5:>8.3f}{log2(6 ** (N // 4)) / N:>14.3f}")
    print(f"{'1:4':<28}{0.25:>8.3f}{log2(4 ** (N // 4)) / N:>14.3f}")
    print(f"{'any 50%, no pattern':<28}{0.5:>8.3f}{log2(comb(N, N // 2)) / N:>14.3f}")


if __name__ == "__main__":
    main()
