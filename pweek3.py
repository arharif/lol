# MENG 404 - Python Week 3
# Required submission filename: pweek3.py

import numpy as np


def ListPrimeNumbers(n):
    # Create the numbers 1 through n.
    numbers = list(range(1, n + 1))

    # Start by assuming every number is prime.
    is_prime = [True] * n

    # 1 is not a prime number.
    if n >= 1:
        is_prime[0] = False

    # Since n <= 400, checking multiples of primes up to 19 is sufficient.
    for p in [2, 3, 5, 7, 11, 13, 17, 19]:
        if p <= n:
            # Keep p itself True and mark only its larger multiples False.
            for multiple in range(2 * p, n + 1, p):
                is_prime[multiple - 1] = False

    # Return the numbers whose corresponding boolean is still True.
    return [
        numbers[i]
        for i in range(n)
        if is_prime[i]
    ]


def Normalize(A):
    # Convert the adjacency matrix to floats.
    B = A.astype(float, copy=True)

    # Divide each column by the sum of that column.
    for j in range(B.shape[1]):
        column_sum = np.sum(B[:, j])

        # If a column contains only zeros, leave it unchanged.
        if column_sum != 0:
            B[:, j] = B[:, j] / column_sum

    return B


def PageRank(A, iter=100):
    # Number of webpages / rows in the normalized matrix.
    n = A.shape[0]

    # The initial rank vector contains 1/n in every entry.
    r = np.full((n, 1), 1.0 / n, dtype=float)

    # The assignment requires len(L) == iter and r0 as the first element.
    if iter <= 0:
        return []

    L = [r.copy()]

    # r_(i+1) = A @ r_i
    for _ in range(iter - 1):
        r = A @ r
        L.append(r.copy())

    return L


def SearchResults(r):
    # Convert the n x 1 rank vector into a one-dimensional array.
    values = np.asarray(r, dtype=float).reshape(-1)

    # Sort indices by rank value from greatest to least.
    order = np.argsort(-values, kind="stable")

    # Return (row_number, rank_value) tuples.
    return [
        (int(i), float(values[i]))
        for i in order
    ]
