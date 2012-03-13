"""
Testing for the nearest centroid module.
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from nose.tools import assert_raises

from sklearn.linear_model import NearestCentroid
from sklearn import datasets

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_classification_toy():
    """Check classification on a toy dataset."""
    clf = NearestCentroid()
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)


def test_iris():
    """Check consistency on dataset iris."""
    for metric in ('euclidean', 'cosine'):
        clf = NearestCentroid().fit(iris.data, iris.target)

        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, "Failed with score = " + str(score)


def test_boston():
    """Check consistency on dataset boston house prices."""
    for metric in ('euclidean', 'cosine'):
        clf = NearestCentroid(metric=metric).fit(iris.data, iris.target)
        score = np.mean(np.power(clf.predict(boston.data) - boston.target, 2))
        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, "Failed with score = " + str(score)


def test_pickle():
    import pickle

    # classification
    obj = NearestCentroid()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(iris.data, iris.target)
    assert score == score2, "Failed to generate same score " + \
            " after pickling (classification) "


if __name__ == "__main__":
    import nose
    nose.runmodule()
