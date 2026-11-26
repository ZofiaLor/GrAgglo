# data sources https://snap.stanford.edu/data/loc-gowalla.html https://snap.stanford.edu/data/loc-brightkite.html

from data_entry import DataEntry
from gr_agglo import GrAgglo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sklearn
import pandas
import argparse
import os


def plot_clustering(data, clusters, size, flip_coords, grayscale):
    data.clusters_number = clusters
    if grayscale:
        colors = [(0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.8, 0.8, 0.8)]
        cmap_name = 'grayscale'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=clusters)
    else:
        cmap = "viridis"
    if (size is None) or (size > data.length):
        data_shortened = data
    else:
        data_shortened = DataEntry(data.data[np.random.choice(data.length, size, replace=False), :], None,
                         "data_shortened", data.dim, data.clusters_number)
    data_shortened.clusters_number = clusters
    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=clusters, linkage='single')
    ac.fit(data_shortened.data)

    plt.figure()
    if flip_coords:
        plt.scatter(data_shortened.data[:, 1], data_shortened.data[:, 0], c=ac.labels_, cmap=cmap, s=6)
    else:
        plt.scatter(data_shortened.data[:, 0], data_shortened.data[:, 1], c=ac.labels_, cmap=cmap, s=6)
    plt.show()

    gr_agglo = GrAgglo(n_clusters=data.clusters_number)
    hc, fcm, granules = gr_agglo.fuzzy_fit(data.data, linkage="single", relation_type='t', xi=0.1)

    plt.figure(figsize=(12, 10))
    if flip_coords:
        plt.scatter(data.data[:, 1], data.data[:, 0], c=(0.95, 0.95, 0.95), s=6)
        plt.scatter(fcm.cluster_centers_[:, 1], fcm.cluster_centers_[:, 0], c=hc.labels, cmap=cmap, s=6)
    else:
        plt.scatter(data.data[:, 0], data.data[:, 1], c=(0.95, 0.95, 0.95), s=6)
        plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=hc.labels, cmap=cmap, s=6)
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Analyze Large Dataset",
                                     description="View results of fuzzy hierarchical clustering on a large 2D dataset")
    parser.add_argument("filename", help="path to the data file")
    parser.add_argument("-c", "--n_clusters", type=int, default=6, help="number of expected clusters")
    parser.add_argument("-s", "--size", type=int, help="the size of data for non-fuzzy clustering")
    parser.add_argument("-f", "--flip_coordinates", action="store_true", help="flip the data's coordinates on plots")
    parser.add_argument("--grayscale", action="store_true", help="plot in grayscale")
    args = parser.parse_args()
    path = args.filename
    if os.path.exists(path):
        data = np.genfromtxt(path, delimiter='\t')
        long_data = DataEntry(data, None, "data", np.shape(data)[1])
        plot_clustering(long_data, args.n_clusters, args.size, args.flip_coordinates, args.grayscale)
    else:
        print("File not found")


if __name__ == "__main__":
    main()

