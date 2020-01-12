# calculate expected mutual information without cython
# https://github.com/jkitzes/batid/blob/master/src/xsklearn/metrics/cluster/supervised.py

from math import log
from scipy.special import gammaln
import numpy as np
import time
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed


def nij_op(
    s1i, s2, l2, N, term1, nijs, i, gln_a, gln_b, gln_Na, gln_Nb, gln_N, gln_nij
):
    emif = 0
    for j in range(l2):
        s2j = s2[j]
        min_nij = np.max([1, s1i + s2j - N])
        max_nij = np.min([s1i, s2j])
        nij = np.arange(min_nij, max_nij) + 1
        t1 = term1[nij]

        t2 = np.log(N * nijs[nij]) - np.log(s1i * s2j)

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


def emi_parallel(contingency, n_samples):
    """
    EMI without pregenerating lookup table for reduced memory
    https://github.com/clajusch/ClEvaR/blob/master/R/Calculations.R
    """

    print("EMI reduced memory parallel")
    s1 = np.array(np.sum(contingency, axis=1, dtype="int").flatten()).flatten()
    s2 = np.array(np.sum(contingency, axis=0, dtype="int").flatten()).flatten()
    N = n_samples
    l1 = len(s1)
    l2 = len(s2)

    nijs = np.arange(0, max(np.max(s1), np.max(s2)) + 1, dtype="float")
    nijs[0] = 1
    term1 = nijs / N

    gln_a = gammaln(s1 + 1)
    gln_b = gammaln(s2 + 1)
    gln_Na = gammaln(N - s1 + 1)
    gln_Nb = gammaln(N - s2 + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)

    with Parallel(n_jobs=-1, verbose=0, prefer=None) as parallel:
        emi = parallel(
            delayed(nij_op)(
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
            )
            for i in tqdm(range(l1), desc="compute emi", mininterval=0.25)
        )

    return np.sum(emi)


"""
def expected_mutual_information(contingency, n_samples, verbose=True):

    if verbose:
        pbar = tqdm(total=5)
        pbar.set_description("prep contingency")

    R, C = contingency.shape
    N = float(n_samples)
    # a = np.sum(contingency, axis=1, dtype="int")
    # b = np.sum(contingency, axis=0, dtype="int")
    a = np.ravel(contingency.sum(axis=1).astype(np.int32, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int32, copy=False))

    if verbose:
        pbar.update(1)
        pbar.set_description("prep term 1")

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype="float")
    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij / N
    term1 = nijs / N

    if verbose:
        pbar.update(1)
        pbar.set_description("prep term 2")

    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    # term2 uses the outer product
    log_ab_outer = np.log(a)[:, np.newaxis] + np.log(b)
    # term2 uses N * nij
    log_Nnij = np.log(N * nijs)

    if verbose:
        pbar.update(1)
        pbar.set_description("prep term 3")

    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(N - a + 1)
    gln_Nb = gammaln(N - b + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)

    if verbose:
        pbar.update(1)
        pbar.set_description("start")

    # start and end values for nij terms for each summation.

    def prep_start(v, N, b):
        return [v - N + w for w in b]

    # start = np.array([prep_start(v, N, b) for v in tqdm(a)], dtype="int")
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        start = np.array(
            parallel(delayed(prep_start)(v, N, b) for v in tqdm(a, leave=False)),
            dtype="int",
        )
    # start = np.array([[v - N + w for w in b] for v in tqdm(a)], dtype="int")
    start = np.maximum(start, 1)

    if verbose:
        pbar.update(1)
        pbar.set_description("end")

    end = np.minimum(np.resize(a, (C, R)).T, np.resize(b, (R, C))) + 1

    pbar.close()

    if verbose:
        print("\n types", {v: [type(i)] for v, i in locals().items()})
        print(
            "sizes",
            {
                v: [str(round(i.nbytes * 1e-9, 3)) + " Gb", np.shape(i)]
                for v, i in locals().items()
                if type(i) == np.ndarray
            },
        )

    # emi itself is a summation over the various values.
    parallel_emi = True
    if parallel_emi:

        if verbose:
            print("sizes", C, R, np.shape(start), np.shape(end))

        flat_iterator = np.concatenate(
            [[[i, j] for j in range(C)] for i in tqdm(range(R), leave=False)]
        )

        def emi_part(
            i,
            j,
            start,
            end,
            term1,
            log_Nnij,
            log_ab_outer,
            gln_a,
            gln_b,
            gln_Na,
            gln_Nb,
            gln_N,
            gln_nij,
            gammaln,
            a,
            b,
        ):
            emii = 0
            for nij in range(start[i, j], end[i, j]):
                term2 = log_Nnij[nij] - log_ab_outer[i, j]
                # Numerators are positive, denominators are negative.
                gln = (
                    gln_a[i]
                    + gln_b[j]
                    + gln_Na[i]
                    + gln_Nb[j]
                    - gln_N
                    - gln_nij[nij]
                    - gammaln(a[i] - nij + 1)
                    - gammaln(b[j] - nij + 1)
                    - gammaln(N - a[i] - b[j] + nij + 1)
                )
                term3 = np.exp(gln)
                # Add the product of all terms.
                emii += term1[nij] * term2 * term3
            return emii

        with Parallel(n_jobs=-1, verbose=0, require="sharedmem") as parallel:
            emi = parallel(
                delayed(emi_part)(
                    i,
                    j,
                    start,
                    end,
                    term1,
                    log_Nnij,
                    log_ab_outer,
                    gln_a,
                    gln_b,
                    gln_Na,
                    gln_Nb,
                    gln_N,
                    gln_nij,
                    gammaln,
                    a,
                    b,
                )
                for i, j in tqdm(flat_iterator, leave=False)
            )
            emi = np.sum(emi)

    else:
        emi = 0
        if verbose:
            itr = tqdm(range(R))
        else:
            itr = range(R)
        for i in itr:
            for j in range(C):
                for nij in range(start[i, j], end[i, j]):
                    term2 = log_Nnij[nij] - log_ab_outer[i, j]
                    # Numerators are positive, denominators are negative.
                    gln = (
                        gln_a[i]
                        + gln_b[j]
                        + gln_Na[i]
                        + gln_Nb[j]
                        - gln_N
                        - gln_nij[nij]
                        - gammaln(a[i] - nij + 1)
                        - gammaln(b[j] - nij + 1)
                        - gammaln(N - a[i] - b[j] + nij + 1)
                    )
                    term3 = np.exp(gln)
                    # Add the product of all terms.
                    emi += term1[nij] * term2 * term3

    if verbose:
        print("ending section emi: {}".format(time.process_time() - start))
        print("emi", emi)
    return emi
    def nij_op(s1i, s2, l2, N):
        emif = 0
        for j in range(l2):
            s2j = s2[j]
            min_nij = np.max([1, s1i + s2j - N])
            max_nij = np.min([s1i, s2j])
            nij = np.arange(min_nij, max_nij) + 1
            t1 = -(nij / N) * np.log((nij * N) / (s1i * s2j))
            t2 = np.exp(
                gammaln(s1i + 1)
                + gammaln(s2j + 1)
                + gammaln(N - s1i + 1)
                + gammaln(N - s2j + 1)
                - gammaln(N + 1)
                - gammaln(nij + 1)
                - gammaln(s1i - nij + 1)
                - gammaln(s2j - nij + 1)
                - gammaln(N - s1i - s2j + nij + 1)
            )
            emi = np.sum(t1 * t2)
            emif += emi
        return emif

    def emi_parallel(contingency, n_samples):
        #
        #EMI without pregenerating lookup table for reduced memory
        #https://github.com/clajusch/ClEvaR/blob/master/R/Calculations.R
        #
        print("EMI reduced memory parallel")
        s1 = np.array(np.sum(contingency, axis=1, dtype="int").flatten()).flatten()
        s2 = np.array(np.sum(contingency, axis=0, dtype="int").flatten()).flatten()
        N = n_samples
        l1 = len(s1)
        l2 = len(s2)
        s_emi = 0

        with Parallel(n_jobs=-1, verbose=0, prefer=None) as parallel:
            emi = parallel(
                delayed(nij_op)(s1[i], s2, l2, N)
                for i in tqdm(range(l1), desc="compute emi", mininterval=0.25)
            )
            s_emi += np.sum(emi)

        return s_emi




def nij_op(s1i, s2j, N):
    min_nij = np.max([1, s1i + s2j - N])
    max_nij = np.min([s1i, s2j])
    nij = range(min_nij, max_nij)
    t1 = -(nij / N) * np.log((nij * N) / (s1i * s2j))
    t2 = np.exp(
        gammaln(s1i)
        + gammaln(s2j)
        + gammaln(N - s1i)
        + gammaln(N - s2j)
        - gammaln(N)
        - gammaln(nij)
        - gammaln(s1i - nij)
        - gammaln(s2j - nij)
        - gammaln(N - s1i - s2j + nij)
    )
    emi = np.sum(t1 * t2)
    return emi

if False:
    emi = parallel(
        delayed(nij_op)(s1[i], s2[j], N)
        for i in tqdm(range(l1), desc="compute emi", mininterval=0.25)
        for j in range(l2)
    )
    s_emi += np.sum(emi)

if True:
    

if False:
    for i in tqdm(range(l1)):
        s1i = s1[i]
        emi = parallel(delayed(nij_op)(s1i, s2[j], N) for j in range(l2))
        s_emi += np.sum(emi)
if False:
    for i in tqdm(range(l1)):
        s1i = s1[i]
        emi = [nij_op(s1i, s2[j], N) for j in range(l2)]
        s_emi += np.sum(emi)
"""

