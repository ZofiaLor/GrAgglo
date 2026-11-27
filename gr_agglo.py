import math
import numpy as np
import sklearn
from fuzzy_cmeans import FuzzyCMeans


class FuzzyNumber:
    """
    A fuzzy number models a number with some tolerance. It is represented by two values:
    a strict number 'x' and fuzziness 'p'. How these values are interpreted depends on
    the shape or relation type of the fuzzy number: triangular, exponential or gaussian.
    A strict number can be modeled as a special case of fuzzy number with p = 0.

    Attributes
    ----------
    x : float
        strict value of the fuzzy number

    p : float
        fuzziness of the fuzzy number
    """

    def __init__(self, x, p):
        """
        Constructs a fuzzy number with a given strict value and fuzziness

        Parameters
        ----------
        x : float
            strict value of the fuzzy number

        p : float
            fuzziness of the fuzzy number
        """
        self.x = x
        self.p = p

    def __repr__(self):
        return f"{self.x}({self.p})"

    def equal_t(self, other):
        """
        Equality operator for triangular fuzzy numbers
        (a == b) === max(1 - |a.x - b.x| / max(a.p, b.p), 0)

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        Returns
        -------
        float
            the value of equality [0.0,1.0]
        """
        return max(1 - (abs(self.x - other.x) / max(self.p, other.p)), 0)

    def equal_e(self, other):
        """
        Equality operator for exponential fuzzy numbers
        (a == b) === exp(-|a.x - b.x| / max(a.p, b.p))

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        Returns
        -------
        float
            the value of equality [0.0,1.0]
        """
        return math.exp(-abs(self.x - other.x) / max(self.p, other.p))

    def equal_g(self, other):
        """
        Equality operator for gaussian fuzzy numbers
        (a == b) === exp((a.x - b.x)^2 / (2 * max(a.p, b.p)^2))

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        Returns
        -------
        float
            the value of equality [0.0,1.0]
        """
        return math.exp(-((self.x - other.x) ** 2) / (2 * (max(self.p, other.p) ** 2)))

    def equal(self, other, relation_type='t'):
        """
        Equality operator for fuzzy numbers. For two strict numbers (a.p == b.p == 0),
        a standard equality operation is performed, returning 1 for true, 0 for false.

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        relation_type: str, default='t'
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        Returns
        -------
        float
            the value of equality [0.0,1.0]
        """
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
        """
        '<' operator for fuzzy numbers.

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        relation_type: str, default='t'
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        Returns
        -------
        float
            the value of the '<' operation [0.0,1.0]
        """
        if self.x > other.x:
            return 0
        return 1 - self.equal(other, relation_type)

    def more(self, other, relation_type='t'):
        """
        '>' operator for fuzzy numbers.

        Parameters
        ----------
        other : FuzzyNumber
            compared fuzzy number

        relation_type: str, default='t'
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        Returns
        -------
        float
            the value of the '>' operation [0.0,1.0]
        """
        if self.x < other.x:
            return 0
        return 1 - self.equal(other, relation_type)

    def add(self, other):
        """
        Addition operator for fuzzy numbers

        Parameters
        ----------
        other : FuzzyNumber
            added fuzzy number

        Returns
        -------
        FuzzyNumber
            result of the addition
        """
        return FuzzyNumber(self.x + other.x, max(self.p, other.p))

    def subtract(self, other):
        """
        Subtraction operator for fuzzy numbers

        Parameters
        ----------
        other : FuzzyNumber
            added fuzzy number

        Returns
        -------
        FuzzyNumber
            result of the subtraction
        """
        return FuzzyNumber(self.x - other.x, max(self.p, other.p))

    def pow(self, power):
        """
        Power operator for fuzzy numbers

        Parameters
        ----------
        power : float
            exponent

        Returns
        -------
        FuzzyNumber
            result of the power operation
        """
        return FuzzyNumber(self.x ** power, self.p)

    def sqrt(self):
        """
        Square root operator for fuzzy numbers

        Returns
        -------
        FuzzyNumber
            result of the extraction
        """
        return FuzzyNumber(math.sqrt(self.x), self.p)


class Granule:
    """
    A multidimensional fuzzy number

    Parameters
    ----------
    center : list(float)
        list of fuzzy number centers (strict 'x' values)

    fuzziness : list(float)
        list of fuzzy number fuzziness values ('p')

    Attributes
    ----------
    fuzzy_dims : list(FuzzyNumber)
        values of each dimension
    """

    def __init__(self, center, fuzziness):
        """
        Constructs a multidimensional fuzzy number

        Parameters
        ----------
        center : list(float)
            list of fuzzy number centers (strict 'x' values)

        fuzziness : list(float)
            list of fuzzy number fuzziness values ('p')
        """
        self.fuzzy_dims = []
        for i in range(len(center)):
            self.fuzzy_dims.append(FuzzyNumber(center[i], fuzziness[i]))

    def fuzzy_distance(self, other):
        """
        Calculate the value of Euclidean fuzzy distance between two granules

        Parameters
        ----------
        other : Granule
            other granule

        Returns
        -------
        FuzzyNumber
            distance between two granules
        """
        fdist = FuzzyNumber(0, 0)
        for i in range(len(self.fuzzy_dims)):
            fdist = fdist.add((self.fuzzy_dims[i].subtract(other.fuzzy_dims[i])).pow(2))
        fdist = fdist.sqrt()
        return fdist


def _single_linkage(left, right, xi, relation_type):
    """
    Chooses the smaller fuzzy value for fuzzy hierarchical clustering.
    If the value of the fuzzy comparison operation is larger than the
    threshold parameter xi, it evaluates to True.

    Parameters
    ----------
    left : FuzzyNumber
        left fuzzy number

    right : FuzzyNumber
        right fuzzy number

    xi : float
        threshold parameter [0.0, 1.0]

    relation_type : str
        relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

    Returns
    -------
    FuzzyNumber
        distance between two granules
    """
    if left.less(right, relation_type) > xi:
        return left
    else:
        return right


def _complete_linkage(left, right, xi, relation_type):
    """
        Chooses the larger fuzzy value for fuzzy hierarchical clustering.
        If the value of the fuzzy comparison operation is larger than the
        threshold parameter xi, it evaluates to True.

        Parameters
        ----------
        left : FuzzyNumber
            left fuzzy number

        right : FuzzyNumber
            right fuzzy number

        xi : float
            threshold parameter [0.0, 1.0]

        relation_type : str
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        Returns
        -------
        FuzzyNumber
            distance between two granules
        """
    if left.more(right, relation_type) > xi:
        return left
    else:
        return right


class HierarchicalClustering:
    """
    General fuzzy and strict hierarchical clustering algorithm

    Parameters
    ----------
    n_clusters : int
        number of expected clusters

    Attributes
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        fuzzy distance matrix

    labels : ndarray of shape (n_samples)
        labels of each granule evaluated by the hierarchical clustering algorithm
    """
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.distances = None
        self.labels = None

    def fuzzy_distance(self, granules):
        """
        Calculate the fuzzy distance matrix - fuzzy distances between each pair of granules.

        Parameters
        ----------
        granules : list(Granule)
            list of granules for clustering

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            fuzzy distance matrix
        """
        n_samples = np.size(granules, 0)
        self.distances = np.full((n_samples, n_samples), FuzzyNumber(np.inf, 0))
        for row in range(n_samples - 1):
            for col in range(row + 1, n_samples):
                self.distances[row, col] = self.distances[col, row] = granules[row].fuzzy_distance(granules[col])
        return self.distances

    def fuzzy_fit(self, granules, xi, relation_type='t', linkage='single'):
        """
        Compute fuzzy hierarchical clustering of granules. Modifies class attribute 'labels'.

        Parameters
        ----------
        granules : list(Granule)
            list of granules for clustering

        xi : float
            threshold parameter [0.0, 1.0]

        relation_type : str, default='t'
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        linkage : str, default='single'
            hierarchical clustering linkage, can be 'single' or 'complete'
        """
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

    def euclidean_distance(self, X):
        """
        Calculate the distance matrix using the Euclidean metric with infinity on the diagonal

        Parameters
        ----------
        X : list(list(float))
            list of data - multidimensional numbers

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            distance matrix with infinity on the diagonal
        """
        self.distances = sklearn.metrics.pairwise_distances(X)
        np.fill_diagonal(self.distances, np.inf)
        return self.distances

    def fit(self, X):
        """
        Compute hierarchical clustering of granules. Modifies class attribute 'labels'.
        A general O(n^3) implementation of the algorithm based on
        https://github.com/hhundiwala/hierarchical-clustering/tree/master. Single linkage only.

        Parameters
        ----------
        X : list(list(float))
            list of data - multidimensional numbers
        """
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


def calculate_fcm_variance(fcm):
    """
    Calculate fuzziness of granules based on FCM algorithm's results.

    Parameters
    ----------
    fcm : FuzzyCMeans
        object of the FuzzyCMeans class with computed clustering

    Returns
    -------
    ndarray of shape (n_granules, n_attributes)
        fuzziness of each cluster in each dimension
    """
    n_granules = fcm.n_clusters
    n_attributes = fcm.X_.shape[1]
    n_samples = fcm.X_.shape[0]
    m = fcm.m
    variances = np.empty(shape=(n_granules, n_attributes))
    for c in range(n_granules):
        for a in range(n_attributes):
            numerator = sum(
                (fcm.U_[i, c] ** m) * (fcm.X_[i, a] - fcm.cluster_centers_[c, a]) ** 2 for i in range(n_samples))
            denominator = sum((fcm.U_[i, c] ** m) for i in range(n_samples))
            variances[c, a] = np.sqrt(numerator / denominator)
    return variances


def fcm_clusters_to_granules(fcm: FuzzyCMeans):
    """
    Convert FuzzyCMeans' clusters into granules - multidimensional fuzzy numbers

    Parameters
    ----------
    fcm : FuzzyCMeans
        object of the FuzzyCMeans class with computed clustering

    Returns
    -------
    granules : list(Granule)
            list of granules
    """
    granules = []
    fuzziness = calculate_fcm_variance(fcm)
    for i in range(fcm.n_clusters):
        granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
    return granules


class GrAgglo:
    """
    Granular agglomerative clustering algorithm. It consists of 3 stages:
    1. Fuzzy C-Means clustering
    2. Conversion of FCM clusters into granules
    3. Fuzzy hierarchical clustering of granules

    Parameters
    ----------
    n_clusters : int, default=3
        number of expected clusters

    fcm_clusters : int, default=100
        number of Fuzzy C-Means clusters, should be a lot smaller than the number of data, but larger than n_clusters

    fcm_max_iter : int, default=80
        maximum number of iterations of the FCM algorithm

    fcm_rng_seed : int, default=None
        random generation seed of FCM algorithm for repeatable results

    Attributes
    ----------
    fcm : FuzzyCMeans
        Fuzzy C-Means object

    hc : HierarchicalClustering
        hierarchical clustering object
    """
    def __init__(self, n_clusters=3, fcm_clusters=100, fcm_max_iter=80, fcm_rng_seed=None):
        self.n_clusters = n_clusters
        self.fcm_clusters = fcm_clusters
        self.fcm_max_iter = fcm_max_iter
        self.fcm_rng_seed = fcm_rng_seed
        self.fcm = FuzzyCMeans(n_clusters=self.fcm_clusters, random_state=self.fcm_rng_seed, max_iter=self.fcm_max_iter)
        self.hc = HierarchicalClustering(n_clusters=self.n_clusters)

    def fuzzy_fit(self, data, linkage="single", relation_type="t", xi=0.5):
        """
        Calculate the GrAgglo algorithm

        Parameters
        ----------
        data : list(list(float))
            list of data - multidimensional numbers

        linkage : str, default='single'
            hierarchical clustering linkage, can be 'single' or 'complete'

        xi : float
            threshold parameter [0.0, 1.0]

        relation_type : str, default='t'
            relation type of the compared numbers chosen among 't' - triangular, 'e' - exponential, 'g' - gaussian

        Returns
        -------
        hc : HierarchicalClustering
            hierarchical clustering object

        fcm : FuzzyCMeans
            Fuzzy C-Means object

        granules : list(Granule)
                list of granules
        """
        self.fcm.fit(data)
        granules = fcm_clusters_to_granules(self.fcm)
        self.hc.fuzzy_fit(granules, xi, relation_type, linkage)
        return self.hc, self.fcm, granules
