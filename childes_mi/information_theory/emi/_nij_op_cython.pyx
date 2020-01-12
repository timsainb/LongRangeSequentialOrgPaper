# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from scipy.special import gammaln
import numpy as np
cimport numpy as np
cimport cython


np.import_array()
ctypedef np.float64_t DOUBLE

def nij_op_cython(int s1i, np.ndarray s2, int l2, int N, np.ndarray term1, 
                  np.ndarray nijs, int i, np.ndarray gln_a, np.ndarray gln_b, 
                  np.ndarray gln_Na, np.ndarray gln_Nb, float gln_N, np.ndarray gln_nij):
    cdef DOUBLE emi
    cdef np.ndarray[DOUBLE] gln, t1, t2, t3
    cdef np.ndarray[np.int32_t] nij 
    cdef int min_nij, max_nij, s2j
    emif = 0
    cdef Py_ssize_t j
    for j in range(l2):
        s2j = s2[j]
        min_nij = max([1, s1i + s2j - N])
        max_nij = min([s1i, s2j])
        nij = np.arange(min_nij, max_nij).astype(np.int32, copy=False) + 1
        t1 = term1[nij]
        t2 = np.log(N * nijs[nij]) - np.log(s1i * s2j)

        gln = (
            gln_a[i] + 
            gln_b[j] + 
            gln_Na[i] + 
            gln_Nb[j] - 
            gln_N - 
            gln_nij[nij] - 
            gammaln(s1i - nij + 1) - 
            gammaln(s2j - nij + 1) -
            gammaln(N - s1i- s2j + nij + 1)
        )

        t3 = np.exp(gln)
        emi = sum(t1 * t2 * t3)
        emif += emi
    return emif