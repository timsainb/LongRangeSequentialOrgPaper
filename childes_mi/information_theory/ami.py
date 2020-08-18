"""
Modified from sklearn
https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/metrics/cluster/_supervised.py

"""
from sklearn.metrics.cluster.supervised import contingency_matrix
import numpy as np
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from .expected_mutual_information import emi_parallel
#from .emi._expected_mutual_info_fast import expected_mutual_information
from .emi._expected_mutual_info_fast_sklearn import expected_mutual_information




def adjusted_mutual_information(labels_true, labels_pred, n_jobs = -1, emi_method="parallel", use_cython=True,
                               average_method='arithmetic'):
    """Adjusted Mutual Information.
    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::
        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
    
    """
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1 or
            classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64,
                                     **_astype_copy_false(contingency))
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    if emi_method == "parallel":
        emi = emi_parallel(contingency, n_samples, use_cython = use_cython, n_jobs=n_jobs)
    else:
        emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo('float64').eps)
    else:
        denominator = max(denominator, np.finfo('float64').eps)
    ami = (mi - emi) / denominator
    return ami, emi


def entropy(labels):
    """Calculates the entropy for a labeling.
    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))

def _generalized_average(U, V, average_method="arithmetic"):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError("'average_method' must be 'min', 'geometric', "
                         "'arithmetic', or 'max'")


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)

import scipy
sp_version = _parse_version(scipy.__version__)

def _astype_copy_false(X):
    """Returns the copy=False parameter for
    {ndarray, csr_matrix, csc_matrix}.astype when possible,
    otherwise don't specify
    """
    if sp_version >= (1, 1) or not sp.issparse(X):
        return {'copy': False}
    else:
        return {}