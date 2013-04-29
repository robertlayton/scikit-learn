# -*- coding: utf-8 -*-
"""
MST: minimal spanning tree with threshold cutting clustering
"""

# Author: Robert Layton <robertlayton@gmail.com>
#
# License: 3-clause BSD.

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree

from ..base import BaseEstimator, ClusterMixin
from ..utils import check_random_state, atleast2d_or_csr


def mst(S, threshold):
    """Compute the Minimum Spanning Tree (MST) on S and then cut at threshold.

    Parameters
    ----------
    S: array [n_samples, n_samples]
        Array of similarities between the samples in the database.
    threshold: float
        Any edge in the MST with a similarity less than this value is
        cut, which separates the tree into sub components.

    Returns
    -------
    labels: array [n_samples]
        An array of label values, the cluster labels for each of the samples
        in the original dataset.

    Notes
    -----
    See examples/plot_eac.py for an example.

    References
    ----------
    Fred, Ana LN, and Anil K. Jain. "Data clustering using evidence
    accumulation." Pattern Recognition, 2002. Proceedings. 16th International
    Conference on. Vol. 4. IEEE, 2002.
    """
    S = atleast2d_or_csr(S)
    n_samples = X.shape[0]
    # Compute Minimum Spanning Tree (currently using Prim's algorithm)
    mst = minimum_spanning_tree(S)
    # Remove any edges less than the threshold
    mst[np.where(S < threshold)] = 0
    # Compute cluster from edges
    labels = _compute_clusters(mst, n_samples)
    return labels


def _compute_clusters(mst):
    """Computes the clusters using the given edges to cluster vertices together

    Parameters
    -------
    mst: array [n_samples, n_samples]
        An array where the value at (i, j) is 0 if samples i and j are not
        connected by an edge and more than zero if they are.

    Returns
    -------
    labels: array [n_samples]
        An array of label values, the cluster labels for each of the samples
        in the original dataset. If any vertice numbers are not represented in
        edges, -1 is given as a label for those, indicating that they are noise
        and not part of any cluster.
    """
    # Each sample in it's own cluster
    labels = np.zeros(mst.shape[0], dtype='int') - 1
    cluster_number = -1  # Label of the next cluster to be found
    # Merge clusters with common edges.
    for (v0, v1) in np.nonzero(mst):
        if labels[v0] == -1 and labels[v1] == -1:
            # Neither sample in a cluster, create a new cluster and add them.
            cluster_number += 1
            labels[v0] = labels[v1] = cluster_number
        elif labels[v0] == -1:
            # Put v0 into v1's cluster
            labels[v0] = labels[v1]
        elif labels[v1] == -1:
            # Put v1 into v0's cluster
            labels[v1] = labels[v0]
        else:
            # All of v1's cluster is renamed with v0's
            labels[labels == labels[v1]] = labels[v0]
    # Rebase the array so there are no "gaps" in the numbering (i.e. 0 to k).
    values = sorted(set(labels))
    # Update each cluster number. Start high so that there are no conflicts.
    for i, value in zip(range(len(values)), values)[::-1]:
        labels[labels == value] = i
    return labels


class MSTThreshold(BaseEstimator, ClusterMixin):
    """Computes the Minimum Spanning Tree (MST) and cuts based on a threshold.

    Parameters
    ----------
    S: array [n_samples, n_samples]
        Array of similarities between the samples in the database.
    threshold: float
        Any edge in the MST with a similarity less than this value is
        cut, which separates the tree into sub components.

    Attributes
    ----------

    `labels_` : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Same as the self.final_clusterer.labels_

    Notes
    -----
    See examples/plot_eac.py for an example.

    References
    ----------
    Fred, Ana LN, and Anil K. Jain. "Data clustering using evidence
    accumulation." Pattern Recognition, 2002. Proceedings. 16th International
    Conference on. Vol. 4. IEEE, 2002.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, S):
        """Computes the MST and cuts based on a threshold.

        Parameters
        ----------
        S: array [n_samples, n_samples]
            Array of similarities between the samples in the database.
        """
        self.labels_ = mst(S, threshold)
        return self
