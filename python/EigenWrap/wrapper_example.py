#!/usr/bin/env python
import eigen
import numpy as np
from scipy import linalg
m = 3
arr = np.random.rand(m,m)

val,vec = eigen.eig(np.dot(arr.T,arr))
val2,vec2 = np.linalg.eig(np.dot(arr.T,arr))

print val
print np.sort(val2)
print vec
print vec2[:,np.argsort(val2)]
