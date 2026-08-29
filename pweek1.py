# Import required modules
import random
import math
from sympy import Symbol, lambdify

# Define the symbolic variable used by SymPy
x = Symbol("x")


def CreateList(n):
    # Create numbers starting from 1 up to 3n + 1
    # with a step of 3: 1, 4, 7, ..., 3n + 1
    return list(range(1, 3 * n + 2, 3))


def ShowConcatenation(n, m):
    # Create the list of even numbers from 4 to 2n + 4
    even_list = list(range(4, 2 * n + 5, 2))

    # Create the list of odd numbers from 501 to 2m + 501
    odd_list = list(range(501, 2 * m + 502, 2))

    # Return the two original lists and their concatenation
    return [even_list, odd_list, even_list + odd_list]


def Sigmoid(x):
    # Calculate the logistic sigmoid function:
    # S(x) = 1 / (1 + e^(-x))

    # Use this form for non-negative x
    if x >= 0:
        return 1 / (1 + math.exp(-x))

    # Equivalent form for negative x to improve numerical stability
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def Av(L):
    # Return None if the list is empty
    if len(L) == 0:
        return None

    # Check that every element in the list is numeric
    for value in L:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

    # Calculate and return the arithmetic mean
    return sum(L) / len(L)


def RandAv(n):
    # Generate a list containing n random integers
    # between 1 and n, inclusive
    L = [random.randint(1, n) for _ in range(n)]

    # The assignment initially asks to print L,
    # but the submission instructions recommend suppressing print statements.
    # print(L)

    # Reuse the Av function to calculate the average
    return Av(L)


def GD(f, x0, n, eta):
    # Compute the symbolic derivative f'(x)
    fp = f.diff(x)

    # Convert the symbolic function and its derivative
    # into callable numerical Python functions
    f = lambdify(x, f)
    fp = lambdify(x, fp)

    # Store the initial value x0 as the first sequence term
    L = [x0]

    # Generate the remaining n - 1 terms using gradient descent:
    # x_(i+1) = x_i - eta * f'(x_i)
    for _ in range(n - 1):
        x_next = L[-1] - eta * fp(L[-1])
        L.append(x_next)

    # Return the first n terms of the gradient descent sequence
    return L
