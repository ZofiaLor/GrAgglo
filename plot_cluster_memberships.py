import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data_entry import DataEntry
from gr_agglo import HierarchicalClustering, Granule, calculate_fcm_variance, FuzzyNumber, GrAgglo
from fuzzy_cmeans import FuzzyCMeans

num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
n_iterations = 80


def plot_membership(data, n_clusters, n_granules, linkage, relation_type, xi, seed, grayscale):
    size = np.shape(data)[0]
    gr_agglo = GrAgglo(n_clusters=n_clusters, fcm_clusters=n_granules, fcm_rng_seed=seed)
    hc, fcm, granules = gr_agglo.fuzzy_fit(data, linkage, relation_type, xi)

    member_cluster = []
    for i in range(n_clusters):
        member_cluster.append([])

    for i in range(size):
        m = []
        for j in range(n_clusters):
            m.append(0)
        for j in range(len(granules)):
            m[hc.labels[j]] += fcm.U_[i][j]
        for j in range(n_clusters):
            member_cluster[j].append(m[j])
    cmap_name = "Greys" if grayscale else "Blues"
    if n_clusters > 3:
        fig, ax = plt.subplots(2, n_clusters // 2)
        fig.set_size_inches(2 * n_clusters, 8)
        for i in range(n_clusters):
            ax[i // 2, i % 2].scatter(data[:, 0], data[:, 1], c=member_cluster[i], cmap=cmap_name)
            ax[i // 2, i % 2].set_title("Cluster " + str(i))
            ax[i // 2, i % 2].set_box_aspect(1)
        plt.show()
    else:
        fig, ax = plt.subplots(1, n_clusters)
        fig.set_size_inches(4 * n_clusters, 4)
        for i in range(n_clusters):
            ax[i].scatter(data[:, 0], data[:, 1], c=member_cluster[i], cmap=cmap_name)
            ax[i].set_title("Cluster " + str(i))
            ax[i].set_box_aspect(1)
        plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Plot Memberships",
                                     description="Plot memberships of points to each hierarchical clustering cluster")
    parser.add_argument("filename", help="name of the data file (e.g. blobs1000)")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("-r", "--relation_type", choices=["t", "e", "g"], default="t", help="relation type triangular (t), exponential (e) or gaussian (g), triangular by default")
    parser.add_argument("-g", "--granule_number", type=int, default=100, help="number of granules, 100 by default")
    parser.add_argument("-x", "--xi",type=float, default=0.1, help="value of xi parameter between 0 and 1, 0.1 by default")
    parser.add_argument("-s", "--seed", type=int, help="random generation seed for the FCM algorithm")
    parser.add_argument("--grayscale", action="store_true", help="plot in grayscale")
    args = parser.parse_args()
    if args.xi < 0 or args.xi > 1:
        print("Invalid xi value")
        return
    data = None
    n_clusters = 0
    for folder in os.scandir("data"):
        for file in os.scandir(folder.path):
            filename = Path(file.path).stem
            if filename != args.filename:
                continue
            full = np.genfromtxt(file.path, delimiter='\t')
            data = full[:, :-1]
            for key, value in num_of_clusters.items():
                if key in filename:
                    n_clusters = value
                    break
    if data is None:
        print("File not found")
        return

    plot_membership(data, n_clusters, args.granule_number, args.linkage, args.relation_type, args.xi, args.seed, args.grayscale)


if __name__ == "__main__":
    main()
