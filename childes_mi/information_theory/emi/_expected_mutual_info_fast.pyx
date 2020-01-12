# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Robert Layton <robertlayton@gmail.com>
#           Corey Lynch <coreylynch9@gmail.com>
# License: BSD 3 clause

from libc.math cimport exp, lgamma
from scipy.special import gammaln
import numpy as np
cimport numpy as np
cimport cython

np.import_array()
ctypedef np.float64_t DOUBLE


def expected_mutual_information(contingency, int n_samples):
    """Calculate the expected mutual information for two labelings."""
    cdef int R, C, a_i, b_j, min_nij, max_nij
    cdef DOUBLE N, gln_N, emi
    cdef np.ndarray[DOUBLE] gln_a, gln_b, gln_Na, gln_Nb, gln_nij, log_Nnij
    cdef np.ndarray[DOUBLE] nijs, term1, term2, gln, term3
    cdef np.ndarray[np.int32_t] a, b, nija
    
    
    R, C = contingency.shape
    N = <DOUBLE>n_samples
    a = np.ravel(contingency.sum(axis=1).astype(np.int32, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int32, copy=False))
    
    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    
    # term1 is nij / N
    term1 = nijs / N

    # term2 uses N * nij
    log_Nnij = np.log(N * nijs)
    
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(N - a + 1)
    gln_Nb = gammaln(N - b + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)
    
    
    # emi itself is a summation over the various values.
    emi = 0
    cdef Py_ssize_t i, j
    for i in range(R):
        a_i = a[i]
        for j in range(C):
            b_j = b[j]
            
            min_nij = int(np.max([1, a_i - N + b_j]))
            max_nij = int(np.min([a_i, b_j]) + 1)
            
            nij = np.arange(min_nij, max_nij).astype(np.int32, copy=False)
            
            term2 = log_Nnij[nij] - np.log(a_i * b_j)
            
            gln = (
                gln_a[i]
                + gln_b[j]
                + gln_Na[i]
                + gln_Nb[j]
                - gln_N
                - gln_nij[nij]
                - lgamma(a[i] - nij + 1)
                - lgamma(b[j] - nij + 1)
                - lgamma(N - a[i] - b[j] + nij + 1)
            )
            
            term3 = exp(gln)
            
            emi += sum(term1[nij] * term2 * term3)
            
    return emi