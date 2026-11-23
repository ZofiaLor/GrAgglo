import matplotlib.pyplot as plt
import sklearn
import numpy as np
from gr_agglo import GrAgglo, HierarchicalClustering, Granule, calculate_fcm_variance, FuzzyNumber, fcm_clusters_to_granules
from fuzzy_cmeans import FuzzyCMeans
import time
import pandas
import os

# Constants
seed = 42
n_iterations = 80


class DataEntry(object):
    def __init__(self, data: np.ndarray, labels: np.ndarray | None = None, name: str = "data", dim: int = 2, clusters_number: int = 3, permute_data: bool = True):
        self.data = data
        self.labels = labels
        self.dim = dim
        self.name = name
        self.length = self.data.shape[0]
        if permute_data:
            rng = np.random.default_rng(seed)
            p = rng.permutation(self.length)
            if self.labels is not None:
                self.data, self.labels = self.data[p], self.labels[p]
            else:
                self.data = self.data[p]
        self.clusters_number = clusters_number
        self.granules_number = [50, 100, 200]
        self.xi = [0.1, 0.5, 0.9]

        self.strict_number_times = []
        self.fuzzy_number_times = dict()

    def measure_strict_number_times(self, repeats, use_sklearn_lib):
        self.strict_number_times = []
        print("Measure strict algorithm")
        for i in range(repeats):
            if use_sklearn_lib:
                hc = sklearn.cluster.AgglomerativeClustering(self.clusters_number)
            else:
                hc = HierarchicalClustering(self.clusters_number)
            start = time.time()
            hc.fit(self.data)
            end = time.time()
            self.strict_number_times.append((end - start) * 1000)

    def measure_fuzzy_number_times(self, repeats):
        self.fuzzy_number_times = dict()
        for n in self.granules_number:
            print("Measure fuzzy algorithm - " + str(n) + " granules")
            for j in range(repeats):
                gr_agglo = GrAgglo(n_clusters=self.clusters_number, fcm_clusters=n, fcm_max_iter=n_iterations, fcm_rng_seed=seed)

                start = time.time()
                gr_agglo.fuzzy_fit(data=self.data)
                end = time.time()
                self.fuzzy_number_times[n] = (end - start) * 1000
                print((end - start) * 1000)

    def measure_times(self, repeats, use_sklearn_lib):
        self.measure_strict_number_times(repeats, use_sklearn_lib)
        self.measure_fuzzy_number_times(repeats)
        print(np.mean(self.strict_number_times))
        for n in self.granules_number:
            print(np.mean(self.fuzzy_number_times[n]))

    def calculate_confusion_matrix(self, granule_member_labels, noise):
        l_matrix = np.zeros(shape=(self.clusters_number, self.clusters_number))
        for d in range(len(granule_member_labels)):
            if granule_member_labels[d] > -1:
                l_matrix[self.labels[d], granule_member_labels[d]] = l_matrix[
                                                                         self.labels[d], granule_member_labels[d]] + 1

        for i in range(self.clusters_number - 1):
            (row, col) = np.unravel_index(np.argmax(l_matrix[i:, i:]),
                                          (self.clusters_number - i, self.clusters_number - i))
            row = row + i
            col = col + i
            l_matrix[[i, row], :] = l_matrix[[row, i], :]
            l_matrix[:, [i, col]] = l_matrix[:, [col, i]]

        sum_diagonal = 0
        recall = 0
        precision = 0
        recall_denominators = np.sum(l_matrix, axis=1).tolist()
        precision_denominators = np.sum(l_matrix, axis=0).tolist()
        for i in range(self.clusters_number):
            sum_diagonal = sum_diagonal + l_matrix[i, i]
            if recall_denominators[i] != 0:
                recall += l_matrix[i, i] / recall_denominators[i]
            if precision_denominators[i] != 0:
                precision += l_matrix[i, i] / precision_denominators[i]
        accuracy = sum_diagonal / (len(granule_member_labels) - noise)
        recall /= self.clusters_number
        precision /= self.clusters_number
        noise_percentage = noise / self.length
        return accuracy, recall, precision, noise_percentage

    def calculate_accuracy(self, fcm, hc, granules, relation_type, shape_dependent_membership):
        granule_member_labels = []
        noise = 0

        if shape_dependent_membership:
            for point in self.data:
                memberships = np.zeros(len(granules))
                for i in range(len(granules)):
                    total = granules[i].fuzzy_dims[0].equal(FuzzyNumber(point[0], 0), relation_type)
                    for j in range(1, len(point)):
                        total *= granules[i].fuzzy_dims[j].equal(FuzzyNumber(point[j], 0), relation_type)
                    memberships[i] = total
                membership = np.argmax(memberships)
                if memberships[membership] > 0:
                    granule_member_labels.append(int(hc.labels[membership]))
                else:
                    noise += 1
                    granule_member_labels.append(-1)
        else:
            for sample in fcm.U_:
                membership = int(np.argmax(sample))
                granule_member_labels.append(int(hc.labels[membership]))

        return self.calculate_confusion_matrix(granule_member_labels, noise)

    def measure_accuracy_const_seed(self, shape_dependent_membership, linkage='single'):
        result_list = []
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            hc = HierarchicalClustering(n_clusters=self.clusters_number)

            fcm.fit(self.data)
            granules = fcm_clusters_to_granules(fcm)
            for relation_type in ['t', 'e', 'g']:
                for k100 in range(5, 100, 5):
                    k = k100 / 100
                    print("granule number " + str(n) + " type " + relation_type + " xi " + str(k))
                    hc.fuzzy_fit(granules, k, relation_type, linkage=linkage)
                    acc, rec, pre, noi = self.calculate_accuracy(fcm, hc, granules, relation_type, shape_dependent_membership)
                    results = {"data size": self.length, "granules number": n, "relation type": relation_type, "xi": k,
                               "accuracy": acc, "recall": rec, "precision": pre, "noise percentage": noi}
                    result_list.append(results)
        return pandas.DataFrame(result_list)

    def measure_accuracy_repeats(self, path, shape_dependent_membership, linkage='single', repeats=10):
        for n in self.granules_number:
            for relation_type in ['t', 'e', 'g']:
                for k100 in range(5, 100, 5):
                    k = k100 / 100
                    print("granule number " + str(n) + " type " + relation_type + " xi " + str(k))
                    accs = []
                    recs = []
                    pres = []
                    nois = []
                    for i in range(repeats):
                        gr_agglo = GrAgglo(n_clusters=self.clusters_number, fcm_clusters=n, fcm_max_iter=n_iterations, fcm_rng_seed=None)
                        hc, fcm, granules = gr_agglo.fuzzy_fit(self.data, linkage=linkage, relation_type=relation_type, xi=k)
                        acc, rec, pre, noi = self.calculate_accuracy(fcm, hc, granules, relation_type, shape_dependent_membership)
                        accs.append(acc)
                        recs.append(rec)
                        pres.append(pre)
                        nois.append(noi)
                    with open(path, 'a') as f:
                        f.write(','.join([repr(self.length), repr(n), relation_type, repr(k), repr(float(sum(accs)/len(accs))), repr(float(sum(recs)/len(recs))), repr(float(sum(pres)/len(pres))), repr(float(sum(nois)/len(nois)))]) + "\n")

    def measure_strict_accuracy(self, linkage="single"):
        result_list = []
        ac = sklearn.cluster.AgglomerativeClustering(n_clusters=self.clusters_number, linkage=linkage)
        ac.fit(self.data)
        acc, rec, pre, noi = self.calculate_confusion_matrix(ac.labels_, 0)
        result_list.append({"data size": self.length, "granules": "none", "accuracy": acc, "recall": rec, "precision": pre})
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            fcm.fit(self.data)
            ac.fit(fcm.cluster_centers_)
            granule_member_labels = []
            for sample in fcm.U_:
                membership = int(np.argmax(sample))
                granule_member_labels.append(int(ac.labels_[membership]))
            acc, rec, pre, noi = self.calculate_confusion_matrix(granule_member_labels, 0)
            result_list.append({"data size": self.length, "granules": str(n), "accuracy": acc, "recall": rec, "precision": pre})
        return pandas.DataFrame(result_list)

    def fit_plot_fuzzy_labels(self, relation_type, linkage="single", file_path: str | None = None, grayscale=False):
        grayscale_shades = [(0.25, 0.25, 0.25), (0.5, 0.5, 0.5), (0.75, 0.75, 0.75)]
        print(self.name)
        labels = []
        centers = []
        fuzz = []
        for n in self.granules_number:
            fcm = FuzzyCMeans(n_clusters=n, random_state=seed, max_iter=n_iterations)
            granules = []

            fcm.fit(self.data)
            centers.append(fcm.cluster_centers_)
            fuzziness = calculate_fcm_variance(fcm)
            for i in range(n):
                granules.append(Granule(fcm.cluster_centers_[i], fuzziness[i]))
            fuzz.append(fuzziness)

            labels_n = []
            for k in self.xi:
                hc = HierarchicalClustering(n_clusters=self.clusters_number)
                hc.fuzzy_fit(granules, k, relation_type, linkage=linkage)
                labels_n.append(hc.labels)
            labels.append(labels_n)

        if self.dim == 2:
            fig, ax = plt.subplots(len(self.granules_number), len(self.xi))
            fig.set_figheight(10)
            fig.set_figwidth(12)

            for i in range(len(self.granules_number)):
                for j in range(len(self.xi)):
                    ax[i, j].set_title(str(self.granules_number[i]) + r" granules, $ \xi $ = " + str(self.xi[j]))
                    t = np.linspace(0, 2 * np.pi)
                    if grayscale:
                        for clust in range(self.granules_number[i]):
                            ax[i, j].plot(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.cos(t),
                                          centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(t),
                                          color=grayscale_shades[labels[i][j][clust]])
                    else:
                        ax[i, j].scatter(self.data[:, 0], self.data[:, 1], c='lightgray')
                        ax[i, j].scatter(centers[i][:, 0], centers[i][:, 1], c=labels[i][j])
                        for clust in range(self.granules_number[i]):
                            ax[i, j].plot(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.cos(t),
                                          centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(t),
                                          color='deeppink', alpha=0.5)
        else:
            fig = plt.figure(figsize=(12, 10))
            for i in range(len(self.granules_number)):
                for j in range(len(self.xi)):
                    t = np.linspace(0, 2 * np.pi, num=10)
                    p = np.linspace(0, np.pi, num=10)
                    t, p = np.meshgrid(t, p)
                    ax = fig.add_subplot(len(self.granules_number), len(self.xi),
                                         i * len(self.granules_number) + j + 1, projection='3d')
                    ax.set_title(str(self.granules_number[i]) + r" granul, $ \xi $ = " + str(self.xi[j]))
                    if grayscale:
                        for clust in range(self.granules_number[i]):
                            ax.plot_surface(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.sin(p) * np.cos(t),
                                            centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(p) * np.sin(t),
                                            centers[i][clust, 2] + 2 * fuzz[i][clust][2] * np.cos(p),
                                            color=grayscale_shades[labels[i][j][clust]])
                    else:
                        ax.scatter3D(centers[i][:, 0], centers[i][:, 1], centers[i][:, 2], c=labels[i][j])
                        for clust in range(self.granules_number[i]):
                            ax.plot_surface(centers[i][clust, 0] + 2 * fuzz[i][clust][0] * np.sin(p) * np.cos(t),
                                            centers[i][clust, 1] + 2 * fuzz[i][clust][1] * np.sin(p) * np.sin(t),
                                            centers[i][clust, 2] + 2 * fuzz[i][clust][2] * np.cos(p),
                                            color='deeppink', alpha=0.5)
        if file_path is not None:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(file_path + "/" + self.name + relation_type + "_" + linkage[0] + ".pdf")
            plt.close()
        else:
            plt.show()
