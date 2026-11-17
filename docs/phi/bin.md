# bin-register

```python
def to_bin(n):
    if n == 0: return 0
    elif n == 1: return 1
    # Quotient (left side) + Remainder (right side)
    else: return to_bin(n // 2) * 10 + to_bin(n % 2)
```

Standard Binary (2^n):
The decimal number 3 in binary is 0011.

- This represents (1 _ 2^1) + (1 _ 2^0) = 2 + 1 = 3

In a hardware sense, a Binary Adder is a masterpiece of efficiency because it uses the absolute minimum number of logic gates to process a carry.

1. The Binary Full Adder (Minimalist)
   A standard binary full adder adds three bits (A, B, and a Carry-in) and outputs a Sum and a Carry-out. It only requires 5 logic gates.

- Logic Gates Used: 2 XOR, 2 AND, 1 OR.
- The "Trick": It only cares about the current column and the one to its immediate left. The carry "ripples" in one direction.

## Equation:

- $Sum = (A \oplus B) \oplus C_{in}$
  In digital electronics and Boolean algebra, the equation $Sum = (A \oplus B) \oplus C_{in}$ translates to:
  "The sum is equal to A exclusive-OR B, exclusive-OR carry-in."
  Key Components

* Sum: The result of the addition operation.
* $\oplus$ (Exclusive-OR / XOR): A logical operation that outputs true (1) only if the number of true inputs is odd.
* $A$ and $B$: The two primary input bits being added.
* $C_{in}$ (Carry-in): The carry bit received from a previous, less significant bit addition.

Context: The Full Adder
This specific formula defines the Sum output of a Full Adder circuit. In this context, the operation can also be described as:

- Parity Check: The sum bit is 1 if an odd number of inputs ($A$, $B$, and $C_{in}$) are 1.
- Modulo-2 Addition: Adding the three bits together and keeping only the remainder after dividing by 2 ($A + B + C_{in} \pmod 2$).

## Carry-out ($C_{out}$) of a full adder is typically expressed in one of two ways:

1. Standard Logic Form
   $C_{out} = (A \cdot B) + (C_{in} \cdot (A \oplus B))$
   In plain English, this translates to:
   "Carry-out is equal to (A AND B) OR (Carry-in AND (A XOR B))."
   This version is most common because it reuses the $A \oplus B$ result already calculated for the Sum, making it efficient to build with physical gates.
2. Simplified (Sum-of-Products) Form
   $C_{out} = AB + BC_{in} + AC_{in}$
   In plain English:
   "Carry-out is equal to (A AND B) OR (B AND Carry-in) OR (A AND Carry-in)."
   What this formula means
   The Carry-out bit is "1" if at least two of the three inputs are "1". [8, 10, 11]

- Generate ($A \cdot B$): If both A and B are 1, they "generate" a carry regardless of what the previous stage sent.
- Propagate ($C_{in} \cdot (A \oplus B)$): If only one of A or B is 1, they "propagate" the incoming carry to the next stage.

The truth table for a full adder shows every possible combination of the three inputs ($A$, $B$, and $C_{in}$) and the resulting Sum and Carry-out ($C_{out}$).

| $A$ | $B$ | $C_{in}$ | Sum ($S$) | Carry-out ($C_{out}$) |
| --- | --- | -------- | --------- | --------------------- |
| 0   | 0   | 0        | 0         | 0                     |
| 0   | 0   | 1        | 1         | 0                     |
| 0   | 1   | 0        | 1         | 0                     |
| 0   | 1   | 1        | 0         | 1                     |
| 1   | 0   | 0        | 1         | 0                     |
| 1   | 0   | 1        | 0         | 1                     |
| 1   | 1   | 0        | 0         | 1                     |
| 1   | 1   | 1        | 1         | 1                     |

How to read this table:

- Sum is 1 whenever there is an odd number of input 1s (1 or 3 total).
- Carry-out is 1 whenever there are two or more input 1s.
- Row 8 (All 1s): When $A$, $B$, and $C_{in}$ are all 1, the total is 3. In binary, 3 is represented as 11, meaning both the Sum and Carry-out bits are 1.

