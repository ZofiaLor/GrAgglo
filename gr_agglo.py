import math
import numpy as np
import sklearn
from fuzzy_cmeans import FuzzyCMeans


class FuzzyNumber:
    def __init__(self, x, p):
        self.x = x
        self.p = p

    def __repr__(self):
        return f"{self.x}({self.p})"

    def equal_t(self, other):
        return max(1 - (abs(self.x - other.x) / max(self.p, other.p)), 0)

    def equal_e(self, other):
        return math.exp(-abs(self.x - other.x) / max(self.p, other.p))

    def equal_g(self, other):
        return math.exp(-((self.x - other.x) ** 2) / (2 * (max(self.p, other.p) ** 2)))

    def equal(self, other, relation_type='t'):
        if self.p == 0 and other.p == 0:
            if self.x == other.x:
                return 1
            return 0
        if relation_type == 't':
            return self.equal_t(other)
        elif relation_type == 'e':
            return self.equal_e(other)
        else:
            return self.equal_g(other)

    def less(self, other, relation_type='t'):
        if self.x > other.x:
            return 0
        return 1 - self.equal(other, relation_type)

    def more(self, other, relation_type='t'):
        if self.x < other.x:
            return 0
        return 1 - self.equal(other, relation_type)

    def add(self, other):
        return FuzzyNumber(self.x + other.x, max(self.p, other.p))

    def subtract(self, other):
        return FuzzyNumber(self.x - other.x, max(self.p, other.p))

    def pow(self, power):
        return FuzzyNumber(self.x ** power, self.p)

    def sqrt(self):
        return FuzzyNumber(math.sqrt(self.x), self.p)


class Granule:
    def __init__(self, center, fuzziness):
        self.fuzzy_dims = []
        for i in range(len(center)):
            self.fuzzy_dims.append(FuzzyNumber(center[i], fuzziness[i]))

    def fuzzy_distance(self, other):
        fdist = FuzzyNumber(0, 0)
        for i in range(len(self.fuzzy_dims)):
            fdist = fdist.add((self.fuzzy_dims[i].subtract(other.fuzzy_dims[i])).pow(2))
        fdist = fdist.sqrt()
        return fdist


def _single_linkage(left, right, xi, relation_type):
    if left.less(right, relation_type) > xi:
        return left
    else:
        return right


def _complete_linkage(left, right, xi, relation_type):
    if left.more(right, relation_type) > xi:
        return left
    else:
        return right


class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.distances = None
        self.labels = None
        self.linkage_matrix = []
        self.linkage_clusters = []

    def fuzzy_distance(self, granules):
        n_samples = np.size(granules, 0)
        self.distances = np.full((n_samples, n_samples), FuzzyNumber(np.inf, 0))
        for row in range(n_samples - 1):
            for col in range(row + 1, n_samples):
                self.distances[row, col] = self.distances[col, row] = granules[row].fuzzy_distance(granules[col])
        return self.distances

    def fuzzy_fit(self, granules, xi, relation_type='t', linkage='single'):
        if xi < 0 or xi > 1:
            print("Xi must have a value between 0 and 1")
            return
        if relation_type not in ['t', 'e', 'g']:
            print("Unsupported relation type")
            return
        formula = None
        if linkage == 'single':
            formula = _single_linkage
        elif linkage == 'complete':
            formula = _complete_linkage
        else:
            print("Invalid linkage")
            return
        self.fuzzy_distance(granules)
        n_samples = np.size(granules, 0)
        self.labels = np.arange(0, n_samples, step=1)
        valid_indices = list(range(n_samples))

        for it in range(n_samples - self.n_clusters):
            min_num = FuzzyNumber(np.inf, 0)
            row = 0
            col = 0
            for i in range(len(valid_indices) - 1):
                for j in range(i + 1, len(valid_indices)):
                    r, c = valid_indices[i], valid_indices[j]
                    if self.distances[r, c].less(min_num, relation_type) > xi:
                        min_num = self.distances[r, c]
                        row = r
                        col = c

            # Iterate over one axis, change the row values to the minimum
            valid_indices.remove(col)
            for i in valid_indices:
                if i != row:
                    min_distance = formula(self.distances[row, i], self.distances[i, col], xi, relation_type)
                    self.distances[i, row] = min_distance
                    self.distances[row, i] = min_distance

            # This does not ensure that the labels are consecutive
            for i in range(n_samples):
                if self.labels[i] == max(row, col):
                    self.labels[i] = min(row, col)

        to_map = np.unique(self.labels)
        for i in range(self.labels.size):
            self.labels[i] = np.where(to_map == self.labels[i])[0][0]

    # Calculate the distance matrix using the Euclidean metric with infinity on the diagonal
    def euclidean_distance(self, X):
        self.distances = sklearn.metrics.pairwise_distances(X)
        np.fill_diagonal(self.distances, np.inf)
        return self.distances

    # Algorithm based on https://github.com/hhundiwala/hierarchical-clustering/tree/master single linkage only
    def fit(self, X):
        self.euclidean_distance(X)
        n_samples = np.size(X, 0)
        self.labels = np.arange(0, n_samples, step=1)

        for it in range(n_samples - self.n_clusters):
            min_distance = np.min(self.distances)  #
            min_unraveled_index = np.argmin(self.distances)
            # The matrix is symmetric so the row/col distinction is not important
            # However, for the later part of the algorithm to work, row < col
            row, col = np.unravel_index(min_unraveled_index, (n_samples, n_samples))
            if row > col:
                row, col = col, row

            # Iterate over one axis, change the row values to the minimum
            for i in range(n_samples):
                if i != row:
                    min_distance = min(self.distances[row, i], self.distances[i, col])
                    self.distances[i, row] = min_distance
                    self.distances[row, i] = min_distance
                self.distances[i, col] = np.inf
                self.distances[col, i] = np.inf

            # This does not ensure that the labels are consecutive
            for i in range(n_samples):
                if self.labels[i] == max(row, col):
                    self.labels[i] = min(row, col)

        to_map = np.unique(self.labels)
        for i in range(self.labels.size):
            self.labels[i] = np.where(to_map == self.labels[i])[0][0]


def calculate_fcm_variance(fuzz, m=2):
    n_granules = fuzz.n_clusters
    n_attributes = fuzz.X_.shape[1]
    n_samples = fuzz.X_.shape[0]
    variances = np.empty(shape=(n_granules, n_attributes))
    for c in range(n_granules):
        for a in range(n_attributes):
            numerator = sum(
                (fuzz.U_[i, c] ** m) * (fuzz.X_[i, a] - fuzz.cluster_centers_[c, a]) ** 2 for i in range(n_samples))
            denominator = sum((fuzz.U_[i, c] ** m) for i in range(n_samples))
            variances[c, a] = np.sqrt(numerator / denominator)
    return variances


def fcm_clusters_to_granules(fcm: FuzzyCMeans):
    granules = []
    fuzziness = calculate_fcm_variance(fcm)
    for i in range(fcm.n_clusters):
        granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
    return granules


class GrAgglo:
    def __init__(self, n_clusters=3, fcm_clusters=100, fcm_max_iter=80, fcm_rng_seed=None):
        self.n_clusters = n_clusters
        self.fcm_clusters = fcm_clusters
        self.fcm_max_iter = fcm_max_iter
        self.fcm_rng_seed = fcm_rng_seed
        self.fcm = FuzzyCMeans(n_clusters=self.fcm_clusters, random_state=self.fcm_rng_seed, max_iter=self.fcm_max_iter)
        self.hc = HierarchicalClustering(n_clusters=self.n_clusters)

    def fuzzy_fit(self, data, linkage="single", relation_type="t", xi=0.5):
        self.fcm.fit(data)
        granules = fcm_clusters_to_granules(self.fcm)
        self.hc.fuzzy_fit(granules, xi, relation_type, linkage)
        return self.hc, self.fcm, granules
