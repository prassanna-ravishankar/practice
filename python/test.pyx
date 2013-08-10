# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc,free
#from libcpp import bool
import numpy as np
cimport numpy as np
np.import_array()
cimport openmp
from cython.parallel import prange
from libc.stdio cimport printf

cdef struct WeakLearner_t:
    int a
    double *b
    int *sharing

class WeakLearner:
    def __init__(self, a, b, sharing):
        self.a = a
        self.b = b
        self.sharing = sharing


cdef class Boosting:
    cdef :
        int size,i,j
        bint *b
        WeakLearner_t *weak
        double error
        np.ndarray img
        np.ndarray arr
    def __cinit__(self):
        self.size = 5
        self.arr = np.zeros(self.size)
        self.weak = <WeakLearner_t *>malloc(sizeof(WeakLearner_t) * self.size)
        self.b = <bint *>malloc(sizeof(bint) * self.size)
        for j in range(self.size):
            self.weak[j].a = 0
            self.weak[j].b = <double *>malloc(sizeof(double) * self.size)
            for i in range(self.size):
                self.weak[j].b[i] = i+j
        
    def __init__(self):
        self.error = 0.5

    def __dealloc__(self):
        for j in range(self.size):
            free(self.weak[j].b)
        free(self.weak)

    def train_round(self, np.ndarray[np.float_t,ndim=1] array):
        cdef int i,n_iter,thread_id,n_thread
        n_iter = array.size
        n_thread = 4
        openmp.omp_set_num_threads(n_thread)
        for i in prange(n_iter, nogil=True):
            thread_id = openmp.omp_get_thread_num()
            array[i] += thread_id
            self.weak[i%self.size].a += thread_id +             self.arr[0]

            printf("%d\n",thread_id)

    def get(self):
        cdef a = self.weak[0].a
        cdef b = self.weak[1].a
        cdef np.npy_intp shape[1]
        shape[0] = self.size
        cdef np.ndarray[np.float_t,ndim=1] sharing = np.PyArray_SimpleNewFromData(1,shape, np.NPY_FLOAT64, self.weak[3].b)
        return WeakLearner(a,b, sharing)
