# MENG 404 - Python Week 5
# Required submission filename: pweek5.py

import numpy as np
from numpy import linalg as LA
from sympy import Symbol, lambdify

x = Symbol("x")
y = Symbol("y")


def FindApproximation(t):
    # Bisection method for x^5 + x + 1 = 0 on [-1, 0].
    a = -1.0
    b = 0.0

    def f(z):
        return z**5 + z + 1

    # Continue until the interval width is less than t.
    while b - a >= t:
        m = (a + b) / 2.0

        # Keep the half-interval where the function changes sign.
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m

    # The assignment asks for the smaller endpoint.
    return min(a, b)


def Proj(P):
    # Projection onto the closed unit ball:
    # D = {(x,y): x^2 + y^2 <= 1}.
    P = np.asarray(P, dtype=float)
    norm = LA.norm(P)

    # If P is already feasible, return it unchanged in value.
    if norm <= 1:
        return P.copy()

    # Otherwise move radially to the unit circle.
    return P / norm


def BoxProj(P):
    # Projection onto the square:
    # S = {(x,y): -1 <= x <= 1 and -1 <= y <= 1}.
    P = np.asarray(P, dtype=float)

    # The closest point is obtained by clipping each coordinate.
    return np.clip(P, -1.0, 1.0)


def PGD(f, P, n, eta, D=1):
    # Symbolic partial derivatives.
    dfx = f.diff(x)
    dfy = f.diff(y)

    # Convert them to numerical functions.
    dfx = lambdify([x, y], dfx)
    dfy = lambdify([x, y], dfy)

    # P0 is the initial point.
    Temp = np.asarray(P, dtype=float).copy()
    L = [Temp.copy()]

    # The examples in the assignment use n projected-gradient updates,
    # producing P0, P1, ..., Pn (n+1 rows).
    for _ in range(n):
        gradient = np.array(
            [dfx(Temp[0], Temp[1]), dfy(Temp[0], Temp[1])],
            dtype=float
        )

        # Ordinary gradient-descent step.
        Temp = Temp - eta * gradient

        # Project the tentative point back into the chosen feasible set.
        if D == 1:
            Temp = Proj(Temp)
        elif D == 2:
            Temp = BoxProj(Temp)
        else:
            raise ValueError("D must be 1 (unit ball) or 2 (square).")

        L.append(Temp.copy())

    return np.array(L)
