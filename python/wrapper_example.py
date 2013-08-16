#!/usr/bin/env python
from EigenWrap import EigenWrap
import numpy as np
from scipy import linalg
m = 10
arr = np.random.rand(m,m)

eigen = EigenWrap()
eigen.set_array(np.dot(arr.T,arr))
v = eigen.eigen_values()
val,vec = np.linalg.eig(np.dot(arr.T,arr))
print v
print np.sort(val)
print np.allclose(v,np.sort(val))
