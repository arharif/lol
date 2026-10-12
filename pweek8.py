# MENG 404 - Python Week 8

import numpy as np


def GiveAGmean(a, b, t):
    """
    Approximate the arithmetic-geometric mean of positive numbers a < b.

    Repeatedly replace:
        a by sqrt(a*b)
        b by (a+b)/2

    using the old values of a and b at each step.  Stop when the
    difference between the two current values is less than t, then
    return the smaller value.
    """
    while abs(b - a) >= t:
        old_a = a
        old_b = b

        a = np.sqrt(old_a * old_b)
        b = (old_a + old_b) / 2

    return min(a, b)


def Centroid(L):
    """
    Return the centroid of the points stored in a 2 x N NumPy array.

    Row 0 contains x-coordinates and row 1 contains y-coordinates.
    The result has shape (2, 1), as required by the assignment.
    """
    return np.mean(L, axis=1).reshape(2, 1)


def CreateCentroids(K):
    """
    Create K deterministic starting centroids for Gradescope testing.

    The fixed seed is required by the assignment so repeated calls
    generate the same starting centroids.
    """
    np.random.seed(30)
    return np.random.randint(0, 101, size=(2, K))


def D(P1, P2):
    """
    Return the squared Euclidean distance between two 2-D points.
    """
    return (P1[0] - P2[0])**2 + (P1[1] - P2[1])**2


def CentroidAssignment(L, C):
    """
    Assign each point in L to its closest centroid in C.

    L has shape (2, N) and C has shape (2, K).  The returned array
    has shape (N,), and each entry is an integer from 0 through K-1.

    np.argmin returns the first minimum, which exactly implements the
    assignment's tie-breaking rule: choose the earliest centroid in C.
    """
    N = L.shape[1]
    K = C.shape[1]

    A = np.zeros(N, dtype=int)

    for i in range(N):
        distances = np.zeros(K)

        for c in range(K):
            distances[c] = D(L[:, i], C[:, c])

        A[i] = np.argmin(distances)

    return A


def NewCentroids(L, A, K):
    """
    Recalculate the K centroids from the current cluster assignments.

    L has shape (2, N), A has shape (N,), and the returned centroid
    array has shape (2, K).  If a cluster has no assigned points,
    NumPy naturally produces NaN for that centroid, matching the
    empty-cluster situation described in the assignment.
    """
    A = np.asarray(A)

    C = np.empty((2, K), dtype=float)

    for c in range(K):
        cluster_points = L[:, A == c]
        C[:, c:c+1] = Centroid(cluster_points)

    return C
