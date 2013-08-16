#!/usr/bin/env python
from EigenWrap import EigenWrap
import numpy as np

m = 10
arr = np.random.rand(m,m)

eigen = EigenWrap()
eigen.set_array(np.dot(arr.T,arr))
val,vec = np.linalg.eig(np.dot(arr.T,arr))
print eigen.eigen_values()
print val
