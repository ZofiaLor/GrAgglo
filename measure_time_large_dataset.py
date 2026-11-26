# data sources https://snap.stanford.edu/data/loc-gowalla.html https://snap.stanford.edu/data/loc-brightkite.html

from data_entry import DataEntry
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas
import argparse
import os

np.random.seed(42)


def measure_time(folderPath, data, clusters, step):
    length = data.length
    r = []
    measured_data = dict()
    if (step is None) or (step > length):
        if length >= 1000000:
            r = [50000, 100000, 500000] + list(range(1000000, length, 1000000))
        else:
            r = list(range(length // 10, length, length // 10))
    else:
        r = list(range(step, length, step))
    for size in r:
        data_shortened = DataEntry(data.data[np.random.choice(length, size, replace=False), :], None,
                                   "data_shortened", data.dim, clusters_number=clusters)
        print(size)
        data_shortened.measure_fuzzy_number_times(1)
        measured_data[size] = dict()
        for n in data_shortened.granules_number:
            print(np.mean(data_shortened.fuzzy_number_times[n]))
            measured_data[size][n] = np.mean(data_shortened.fuzzy_number_times[n])

    s = len(r)
    granule50_avg = np.empty(shape=s)
    granule100_avg = np.empty(shape=s)
    granule200_avg = np.empty(shape=s)

    for i in range(s):
        granule50_avg[i] = np.mean(measured_data[r[i]][50]) // 1000
        granule100_avg[i] = np.mean(measured_data[r[i]][100]) // 1000
        granule200_avg[i] = np.mean(measured_data[r[i]][200]) // 1000

    results = [granule50_avg, granule100_avg, granule200_avg]
    results = [list(i) for i in zip(*results)]
    results = pandas.DataFrame(results, columns=["50 granules", "100 granules", "200 granules"])
    print(results.to_string())
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    if not os.path.isfile(folderPath + "/large_data_times.csv"):
        results.to_csv(folderPath + "/large_data_times.csv")
    else:
        counter = 1
        while os.path.isfile(folderPath + "/large_data_times" + str(counter) + ".csv"):
            counter += 1
        results.to_csv(folderPath + "/large_data_times" + str(counter) + ".csv")

    plt.figure()
    plt.plot(r, granule50_avg)
    plt.plot(r, granule100_avg)
    plt.plot(r, granule200_avg)
    plt.xlabel("Number of data")
    plt.ylabel("Execution time [s]")
    plt.title("Average time of GrAgglo algorithm execution depending on data number")
    legend = plt.legend(["Fuzzy agglomerative clustering of 50 granules",
                         "Fuzzy agglomerative clustering of 100 granules", "Fuzzy agglomerative clustering of 200 granules"],
                        bbox_to_anchor=(0.5, -0.38), loc='lower center')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Analyze Large Dataset",
                                     description="Measure execution time of fuzzy hierarchical clustering on a large dataset")
    parser.add_argument("filename", help="path to the data file")
    parser.add_argument("-c", "--n_clusters", type=int, default=6, help="number of expected clusters")
    parser.add_argument("-s", "--step", type=int, help="the step size of consecutive data measurements")
    parser.add_argument("-o", "--output", default="results/time", help="output folder of result .csv files, results/time by default")
    args = parser.parse_args()
    path = args.filename
    if os.path.exists(path):
        data = np.genfromtxt(path, delimiter='\t')
        long_data = DataEntry(data, None, "data", np.shape(data)[1])
        measure_time(args.output, long_data, args.n_clusters, args.step)
    else:
        print("File not found")


if __name__ == "__main__":
    main()
