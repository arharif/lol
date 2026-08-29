import random
import math
from sympy import Symbol, lambdify

x = Symbol("x")


def CreateList(n):
    return list(range(1, 3 * n + 2, 3))


def ShowConcatenation(n, m):
    even_list = list(range(4, 2 * n + 5, 2))
    odd_list = list(range(501, 2 * m + 502, 2))
    return [even_list, odd_list, even_list + odd_list]


def Sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def Av(L):
    if len(L) == 0:
        return None

    for value in L:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

    return sum(L) / len(L)


def RandAv(n):
    L = [random.randint(1, n) for _ in range(n)]
    # print(L)  # Uncomment only if your instructor wants the Section 2.5 print behavior.
    return Av(L)


def GD(f, x0, n, eta):
    fp = f.diff(x)
    f = lambdify(x, f)
    fp = lambdify(x, fp)

    L = [x0]

    for _ in range(n - 1):
        x_next = L[-1] - eta * fp(L[-1])
        L.append(x_next)

    return L
