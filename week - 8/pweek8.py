import numpy as np


def GiveAGmean(a, b, t):
  # Update the arithmetic and geometric means using the same old
  # values of a and b at each iteration.
  while b - a >= t:
    old_a = a
    old_b = b
    a = np.sqrt(old_a * old_b)
    b = (old_a + old_b) / 2

  # Stop when the difference is less than t and return the smaller one.
  return min(a, b)


def D(P1, P2):
  return (P1[0]-P2[0])**2 + (P1[1]-P2[1])**2


def Centroid(T):
  # The first row stores x-coordinates and the second row stores
  # y-coordinates.  The required output shape is (2, 1).
  return np.mean(T, axis=1).reshape(2, 1)


def CreateCentroids(L, K):
  XMin = int(np.min(L[0]))
  XMax = int(np.max(L[0]))
  YMin = int(np.min(L[1]))
  YMax = int(np.max(L[1]))

  # The professor requires this seed in the Gradescope submission.
  np.random.seed(30)
  X = np.random.randint(XMin, XMax, size=(K))
  Y = np.random.randint(YMin, YMax, size=(K))

  # All testing print statements are suppressed for submission.
  return np.vstack((X, Y))


def CentroidAssignment(L, C):
  # L has shape (2, N), C has shape (2, K), and the result
  # must have shape (N,).
  N = L.shape[1]
  K = C.shape[1]
  A = np.empty(N, dtype=int)

  # Compute the squared distance from each point to every centroid.
  # np.argmin returns the earliest centroid when there is a tie.
  for i in range(N):
    distances = np.empty(K)

    for c in range(K):
      distances[c] = D(L[:, i], C[:, c])

    A[i] = np.argmin(distances)

  return A


def NewCentroids(L, A, K):
  # Boolean indexing is used to collect all points assigned to each
  # cluster and then compute that cluster's centroid.
  A = np.asarray(A)
  C = np.empty((2, K), dtype=float)

  for c in range(K):
    cluster = L[:, A == c]
    C[:, c:c+1] = Centroid(cluster)

  return C
