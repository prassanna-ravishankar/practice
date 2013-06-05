import numpy as np
import convolve_py
import convolve1

N = 100
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))
convolve1.naive_convolve(f, g)
