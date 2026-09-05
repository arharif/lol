# MENG 404 - Python Week 2
# Required submission filename: pweek2.py

import fractions
import numpy as np


def ReturnKthIterate(k):
    # Start with x_0 = 2 and keep exact rational arithmetic.
    x = fractions.Fraction(2, 1)

    # Apply the recurrence k times to obtain x_k.
    for _ in range(k):
        x = fractions.Fraction(1, 2) * (
            x + fractions.Fraction(2, 1) / x
        )

    return x


def SwapRows(A, L):
    # Work on a copy so the original matrix is not changed.
    B = A.copy()

    # Exchange rows L[0] and L[1].
    B[[L[0], L[1]]] = B[[L[1], L[0]]]

    return B


def MulRow(A, r, c):
    # Use float values so fractional row operations are preserved.
    B = A.astype(float, copy=True)

    # Multiply row r by the non-zero constant c.
    B[r] = c * B[r]

    return B


def AddMul(A, L, c):
    # Use float values so fractional row operations are preserved.
    B = A.astype(float, copy=True)

    # Replace row L[1] with row L[1] + c * row L[0].
    B[L[1]] = B[L[1]] + c * B[L[0]]

    return B


def Pivot(A, L):
    # Pivoting can create fractions, so use floating-point values.
    B = A.astype(float, copy=True)

    pivot_row = L[0]
    pivot_col = L[1]
    pivot_value = B[pivot_row, pivot_col]

    # The assignment requires 0 if the selected pivot element is zero.
    if pivot_value == 0:
        return 0

    # Make the pivot equal to 1.
    B[pivot_row] = B[pivot_row] / pivot_value

    # Make every other entry in the pivot column equal to 0.
    for row in range(B.shape[0]):
        if row != pivot_row:
            multiplier = B[row, pivot_col]
            B[row] = B[row] - multiplier * B[pivot_row]

    return B


def rref(A):
    # Work on a floating-point copy for row division.
    B = A.astype(float, copy=True)

    rows, cols = B.shape
    pivot_row = 0
    rank = 0

    # Search for pivots from left to right.
    for col in range(cols):
        if pivot_row >= rows:
            break

        selected_row = None

        # Find a non-zero entry at or below the current pivot row.
        for row in range(pivot_row, rows):
            if B[row, col] != 0:
                selected_row = row
                break

        # No pivot in this column.
        if selected_row is None:
            continue

        # Move the selected row into the pivot position.
        if selected_row != pivot_row:
            B = SwapRows(B, [pivot_row, selected_row])

        # Reduce the pivot column.
        B = Pivot(B, [pivot_row, col])

        # Rank is counted manually from the number of pivots.
        rank += 1
        pivot_row += 1

    # Remove tiny floating-point artifacts.
    B[np.abs(B) < 1e-12] = 0.0

    return B, rank


def PoolMatrix(A, f, type=0):
    # For stride 1, output width/height is n - f + 1.
    n = A.shape[0]
    output_size = n - f + 1

    if type == 0:
        # Max pooling.
        result = np.empty((output_size, output_size), dtype=A.dtype)

        for i in range(output_size):
            for j in range(output_size):
                window = A[i:i + f, j:j + f]
                result[i, j] = np.max(window)

        return result

    if type == 1:
        # Average pooling.
        result = np.empty((output_size, output_size), dtype=float)

        for i in range(output_size):
            for j in range(output_size):
                window = A[i:i + f, j:j + f]
                result[i, j] = np.mean(window)

        return result

    return None
