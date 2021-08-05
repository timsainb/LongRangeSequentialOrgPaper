# calculate expected mutual information in parallel with / without cython
# https://github.com/jkitzes/batid/blob/master/src/xsklearn/metrics/cluster/supervised.py

from math import log
from scipy.special import gammaln
import numpy as np
import time
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed, parallel_backend
from childes_mi.information_theory.emi._nij_op_cython import nij_op_cython


def nij_op(
    s1i,
    s2,
    l2,
    N,
    term1,
    nijs,
    i,
    gln_a,
    gln_b,
    gln_Na,
    gln_Nb,
    gln_N,
    gln_nij,
    log_Nnij,
):
    emif = 0
    for j in range(l2):
        s2j = s2[j]
        min_nij = np.max([1, s1i - N + s2j])
        max_nij = np.min([s1i, s2j]) + 1
        nij = np.arange(min_nij, max_nij)
        t1 = term1[nij]
        t2 = log_Nnij[nij] - np.log(s1i * s2j)

        gln = (
            gln_a[i]
            + gln_b[j]
            + gln_Na[i]
            + gln_Nb[j]
            - gln_N
            - gln_nij[nij]
            - gammaln(s1i - nij + 1)
            - gammaln(s2j - nij + 1)
            - gammaln(N - s1i - s2j + nij + 1)
        )

        t3 = np.exp(gln)
        emi = sum(t1 * t2 * t3)
        emif += emi
    return emif


def emi_parallel(contingency, n_samples, use_cython=True, n_jobs=-1, prefer=None):
    """
    EMI without pregenerating lookup table for reduced memory
    https://github.com/clajusch/ClEvaR/blob/master/R/Calculations.R
    """

    s1 = np.array(np.sum(contingency, axis=1, dtype="int").flatten()).flatten()
    s2 = np.array(np.sum(contingency, axis=0, dtype="int").flatten()).flatten()
    N = n_samples
    l1 = len(s1)
    l2 = len(s2)

    nijs = np.arange(0, max(np.max(s1), np.max(s2)) + 1, dtype="float")
    nijs[0] = 1
    term1 = nijs / N

    log_Nnij = np.log(N * nijs)
    gln_a = gammaln(s1 + 1)
    gln_b = gammaln(s2 + 1)
    gln_Na = gammaln(N - s1 + 1)
    gln_Nb = gammaln(N - s2 + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)

    if use_cython:
        nij_func = nij_op_cython
    else:
        nij_func = nij_op
    with parallel_backend("multiprocessing"):
        with Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer) as parallel:
            emi = parallel(
                delayed(nij_func)(
                    s1[i],
                    s2,
                    l2,
                    N,
                    term1,
                    nijs,
                    i,
                    gln_a,
                    gln_b,
                    gln_Na,
                    gln_Nb,
                    gln_N,
                    gln_nij,
                    log_Nnij,
                )
                for i in tqdm(range(l1), desc="compute emi", leave=False)
            )

    return np.sum(emi)
