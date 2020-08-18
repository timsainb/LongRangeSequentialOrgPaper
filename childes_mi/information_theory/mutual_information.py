# mutual information
import numpy as np
import scipy
from tqdm.autonotebook import tqdm
import scipy.special
from sklearn.metrics.cluster.supervised import contingency_matrix
from scipy import sparse as sp
import copy
from childes_mi.information_theory.ami import adjusted_mutual_information

# modelling
from joblib import Parallel, delayed

from sklearn.metrics.cluster.expected_mutual_info_fast import (
    expected_mutual_information,
)

from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)

######## Estimate Mutual Information Faster ############


def est_mutual_info_p(a, b):
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)

    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))  # number of A
    S_a, var_a = entropyp(Na)  # entropy with P(A) as input

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb)

    # joint
    S_ab, var_ab = entropyp(Nall)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty


######## Mutual Information Faster ############
def entropyp(Nall):
    N = np.sum(Nall)
    pAll = np.array([float(Ni) * scipy.special.psi(float(Ni)) for Ni in Nall])
    S_hat = np.log2(N) - 1.0 / N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype="float32")))

    return S_hat, var


def mutual_info_p(a, b):
    """ Fast mutual information calculation based upon sklearn,
    but with estimation of uncertainty from Lin & Tegmark 2016
    """
    # contingency matrix of a * b
    contingency = contingency_matrix(a, b, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)
    N = len(a)
    # entropy of a
    Na = np.ravel(contingency.sum(axis=0))
    S_a, var_a = entropyp(Na / np.sum(Na))

    # entropy of b
    Nb = np.ravel(contingency.sum(axis=1))
    S_b, var_b = entropyp(Nb / np.sum(Nb))

    # joint entropy
    S_ab, var_ab = entropyp(Nall / N)

    # mutual information
    MI = S_a + S_b - S_ab

    # uncertainty and variance of MI
    MI_var = var_a + var_b + var_ab

    uncertainty = np.sqrt((MI_var) / len(a))

    return MI, uncertainty




######## Mutual Information From distributions ############
def MI_from_distributions(
    sequences,
    dist,
    unclustered_element=None,
    use_sklearn=True,
    n_jobs = -1,
    shuffle = False,
    mi_estimation = "grassberger", # "adjusted_mi", None
    **mi_kwargs
):
    np.random.seed()  # set seed
    # create distributions
    if np.sum([len(seq) > dist for seq in sequences]) == 0:
        return (np.nan, np.nan)

    if shuffle:
        distribution_a = np.concatenate(
            [np.random.permutation(seq[dist:]) for seq in sequences if len(seq) > dist]
        )

        distribution_b = np.concatenate(
            [np.random.permutation(seq[:-dist]) for seq in sequences if len(seq) > dist]
        )
    
    else:
        distribution_a = np.concatenate(
            [seq[dist:] for seq in sequences if len(seq) > dist]
        )

        distribution_b = np.concatenate(
            [seq[:-dist] for seq in sequences if len(seq) > dist]
        )

    # mask unclustered so they are not considered in MI
    if unclustered_element is not None:
        mask = (distribution_a == unclustered_element) | (
            distribution_b == unclustered_element
        )
        distribution_a = distribution_a[mask == False]
        distribution_b = distribution_b[mask == False]

    # calculate MI

    if mi_estimation == "grassberger":
        # See Grassberger, P. Entropy estimates from insufficient samplings. arXiv 2003, arXiv:0307138
        return est_mutual_info_p(distribution_a, distribution_b)
    elif mi_estimation == "adjusted_mi_parallel":
        # parallelized mi estimation for very large distributions
        # See Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance, JMLR
        return adjusted_mutual_information(distribution_a, distribution_b, n_jobs=n_jobs, **mi_kwargs)
    elif mi_estimation == "adjusted_mi_sklearn":
        # modified version of sklearn ami calculation in 0(n) memory instead of O^n
        # See Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance, JMLR
        return adjusted_mutual_information(distribution_a, distribution_b, emi_method="sklearn", **mi_kwargs)
    elif mi_estimation is None:
        # sklearns mi implementation
        return (mutual_info_score(distribution_a, distribution_b, **mi_kwargs), 0)
    else:
        raise ValueError("MI estimator '{}' is not implemented".format(mi_estimation))

def sequential_mutual_information(
    sequences,
    distances,
    n_jobs=1,
    verbosity=5,
    n_shuff_repeats=1, # how many times to shuffle distribution in order to estimate lower bound
    disable_tqdm=False,
    prefer=None,  # is None better here?
    mi_estimation = "grassberger",
    unclustered_element=None,
    **mi_kwargs
):
    """
    Compute mutual information as a function of distance between sequences
    if n_jobs > 1,  will run in parallel
    """
    # convert to numeric for faster computation
    unique_elements = np.unique(np.concatenate(sequences))
    n_unique = len(unique_elements)
    seq_dict = {j: i for i, j in enumerate(unique_elements)}
    if n_unique < 256:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint8") for seq in sequences
        ]
    elif n_unique < 65535:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint16") for seq in sequences
        ]
    else:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint32") for seq in sequences
        ]
    
    # adjust dataset so that unclustered elements are not factored into MI elements
    if unclustered_element is not None:
        unclustered_element = seq_dict[unclustered_element]
        print(unclustered_element)
    else:
        unclustered_element = None

    # because parallelization occurs within the function
    if mi_estimation == "adjusted_mi":    
        _n_jobs = copy.deepcopy(n_jobs)
        n_jobs = 1
    else:
        _n_jobs = 1
    # compute MI
    if n_jobs == 1:
        MI = [
            MI_from_distributions(
                sequences,
                dist,
                unclustered_element=unclustered_element,
                mi_estimation=mi_estimation,
                **mi_kwargs
            )
            for dist_i, dist in enumerate(
                tqdm(distances, leave=False, disable=disable_tqdm)
            )
        ]
        distances_rep = np.repeat(distances, n_shuff_repeats)
        shuff_MI = [
            MI_from_distributions(
                sequences,
                dist,
                unclustered_element=unclustered_element,
                mi_estimation=mi_estimation,
                shuffle=True,
                **mi_kwargs
            )
            for dist_i, dist in enumerate(
                tqdm(distances_rep, leave=False, disable=disable_tqdm)
            )
        ]

        shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
        shuff_MI = np.mean(shuff_MI, axis=1)

    else:
        with Parallel(n_jobs=n_jobs, verbose=verbosity, prefer=prefer) as parallel:
            MI = parallel(
                delayed(MI_from_distributions)(
                    sequences,
                    dist,
                    unclustered_element=unclustered_element,
                    n_jobs = _n_jobs,
                    mi_estimation=mi_estimation,
                    **mi_kwargs
                )
                for dist_i, dist in enumerate(
                    tqdm(distances, leave=False, disable=disable_tqdm)
                )
            )

        with Parallel(n_jobs=n_jobs, verbose=verbosity, prefer=prefer) as parallel:
            distances_rep = np.repeat(distances, n_shuff_repeats)
            shuff_MI = parallel(
                delayed(MI_from_distributions)(
                    sequences,
                    dist,
                    unclustered_element=unclustered_element,
                    n_jobs = _n_jobs,
                    mi_estimation=mi_estimation,
                    shuffle=True,
                    **mi_kwargs
                )
                for dist_i, dist in enumerate(
                    tqdm(distances_rep, leave=False, disable=disable_tqdm)
                )
            )
            shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
            shuff_MI = np.mean(shuff_MI, axis=1)

    return np.array(MI).T, np.array(shuff_MI).T
