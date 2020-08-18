from math import log
from scipy.special import gammaln
import numpy as np
from tqdm.autonotebook import tqdm


def expected_mutual_information_numpy(contingency, n_samples):
    """Calculate the expected mutual information for two labelings.
    This replaces the O^2
    """
    R, C = contingency.shape
    N = float(n_samples)
    a = np.ravel(contingency.sum(axis=1).astype(np.int32, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int32, copy=False))
    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype="float")
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
    for i in range(R):
        a_i = a[i]
        for j in range(C):
            b_j = b[j]
            
            min_nij = int(np.max([1, a_i - N + b_j]))
            max_nij = int(np.min([a_i, b_j]) + 1)
            nij = range(min_nij, max_nij)
            
            # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
            term2 = log_Nnij[nij] - np.log(a_i * b_j)
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
            emi += sum(term1[nij] * term2 * term3)

    
    return emi



def emi(contingency, n_samples):
    """
    EMI without pregenerating lookup table for reduced memory
    https://github.com/clajusch/ClEvaR/blob/master/R/Calculations.R
    """
    print("EMI reduced memory")
    s1 = np.array(np.sum(contingency, axis=1, dtype="int").flatten()).flatten()
    s2 = np.array(np.sum(contingency, axis=0, dtype="int").flatten()).flatten()
    N = n_samples
    l1 = len(s1)
    l2 = len(s2)
    s_emi = 0
    for i in tqdm(range(l1)):
        s1i = s1[i]
        for j in range(l2):
            s2j = s2[j]
            min_nij = np.max([1, s1i + s2j - N])
            max_nij = np.min([s1i, s2j])
            # range from min_nij to max_nij
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
            s_emi += emi
    return emi
