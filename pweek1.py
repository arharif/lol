# Rafia Haqque
# MENG 404 : Python Week 1

import random
import math
from sympy import Symbol, lambdify

# Symbolic variable used by SymPy in the GD function
x = Symbol("x")


def CreateList(n):
    # Numbers from 1 through 3n + 1 with remainder 1 when divided by 3
    return list(range(1, 3 * n + 2, 3))


def ShowConcatenation(n, m):
    # Even numbers from 4 through 2n + 4
    even_list = list(range(4, 2 * n + 5, 2))
    # Odd numbers from 501 through 2m + 501
    odd_list = list(range(501, 2 * m + 502, 2))
    # Return both lists and their concatenation
    return [even_list, odd_list, even_list + odd_list]


def Sigmoid(x):
    # Logistic sigmoid function: 1 / (1 + e^(-x))
    return 1 / (1 + math.exp(-x))


def Av(L):
    # Return None for an empty list
    if len(L) == 0:
        return None
    # Return None if any element is not numeric
    for value in L:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    # Return the arithmetic mean
    return sum(L) / len(L)


def RandAv(n):
    # Create n random integers between 1 and n, inclusive
    L = [random.randint(1, n) for _ in range(n)]
    # Return the average using the Av function
    return Av(L)


def GD(f, x0, n, eta):
    # Compute the symbolic derivative f'(x)
    fp = f.diff(x)
    # Convert the symbolic expressions to callable numerical functions
    f = lambdify(x, f)
    fp = lambdify(x, fp)
    # x0 is the first term of the sequence
    L = [x0]
    # Gradient descent update:
    # x_(i+1) = x_i - eta * f'(x_i)
    for _ in range(n - 1):
        x_next = L[-1] - eta * fp(L[-1])
        L.append(x_next)

    return L
