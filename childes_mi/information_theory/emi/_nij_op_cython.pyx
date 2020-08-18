# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp, lgamma
from scipy.special import gammaln
import numpy as np
cimport numpy as np
cimport cython


np.import_array()
ctypedef np.float64_t DOUBLE

def nij_op_cython(int s1i, np.ndarray s2, int l2, int N, np.ndarray term1, 
                  np.ndarray nijs, int i, np.ndarray gln_a, np.ndarray gln_b, 
                  np.ndarray gln_Na, np.ndarray gln_Nb, float gln_N, np.ndarray gln_nij,
                  np.ndarray log_Nnij ######### create ###########
                 ):
    cdef DOUBLE emi, log_aibj, term3, gln, term2
    cdef int min_nij, max_nij, s2j
    emi = 0
    cdef Py_ssize_t j, nij
    for j in range(l2):
        s2j = s2[j]
        min_nij = max([1, s1i + s2j - N])
        max_nij = min([s1i, s2j]) + 1
        
        log_aibj = np.log(s1i * s2j)
        for nij in range(min_nij, max_nij):
            term2 = log_Nnij[nij] - log_aibj
             
            gln = (
                    gln_a[i]
                    + gln_b[j]
                    + gln_Na[i]
                    + gln_Nb[j]
                    - gln_N
                    - gln_nij[nij]
                    - lgamma(s1i - nij + 1)
                    - lgamma(s2j - nij + 1)
                    - lgamma(N - s1i - s2j + nij + 1)
            )
               
            term3 = exp(gln)
            emi += (term1[nij] * term2 * term3)
        
    return emi