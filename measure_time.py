import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import pandas
from data_entry import DataEntry
import argparse

# Constants
# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]

fullData = {}
names = []


def compare_time_complexity(resultsFolderPath, imageFolderPath, repeats, save_results, save_plot, use_sklearn_lib, name="corners"):
    measured_data = {}
    if use_sklearn_lib:
        suffix = "_sklearn"
        strict_label = "Strict agglomerative clustering with scikit-learn's algorithm"
        data_size_presets_for_time_measurement = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    else:
        suffix = ""
        strict_label = "Strict agglomerative clustering with the general algorithm"
        data_size_presets_for_time_measurement = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for i in data_size_presets_for_time_measurement:
        if use_sklearn_lib:
            measured_data[name + str(i)] = fullData[name + str(i)]
        else:
            measured_data[name + str(i)] = DataEntry(fullData[name + "5000"].data[:i], None, name + str(i), fullData[name + "5000"].dim)
        print("Measure ", name, str(i))
        measured_data[name + str(i)].measure_times(repeats, use_sklearn_lib)

    s = len(data_size_presets_for_time_measurement)
    regular_avg = np.empty(shape=s)
    granule50_avg = np.empty(shape=s)
    granule100_avg = np.empty(shape=s)
    granule200_avg = np.empty(shape=s)

    for i in range(s):
        regular_avg[i] = np.mean(measured_data[name + str(data_size_presets_for_time_measurement[i])].strict_number_times)
        granule50_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[50])
        granule100_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[100])
        granule200_avg[i] = np.mean(
            measured_data[name + str(data_size_presets_for_time_measurement[i])].fuzzy_number_times[200])

    results = [regular_avg, granule50_avg, granule100_avg, granule200_avg]
    results = [list(i) for i in zip(*results)]
    results = pandas.DataFrame(results, columns=["strict", "50 granules", "100 granules", "200 granules"])
    if save_results:
        if not os.path.exists(resultsFolderPath):
            os.makedirs(resultsFolderPath)
        results.to_csv(resultsFolderPath + "/times" + suffix + ".csv")
    else:
        print(results.to_string())

    plt.figure()
    plt.plot(data_size_presets_for_time_measurement, regular_avg)
    plt.plot(data_size_presets_for_time_measurement, granule50_avg)
    plt.plot(data_size_presets_for_time_measurement, granule100_avg)
    plt.plot(data_size_presets_for_time_measurement, granule200_avg)
    plt.xlabel("Number of data")
    plt.ylabel("Execution time [ms]")
    plt.title("Average algorithms execution time depending on data number")
    legend = plt.legend([strict_label, "Fuzzy agglomerative clustering of 50 granules",
                         "Fuzzy agglomerative clustering of 100 granules", "Fuzzy agglomerative clustering of 200 granules"],
                        bbox_to_anchor=(0.5, -0.38), loc='lower center')
    if save_plot:
        if not os.path.exists(imageFolderPath):
            os.makedirs(imageFolderPath)
        plt.savefig(imageFolderPath + "/comparing_times" + suffix + ".pdf", bbox_extra_artists=[legend], bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Measure Time",
                                     description="Measure time of fuzzy hierarchical clusterization in comparison with strict clusterization")
    parser.add_argument("-r", "--repeats", type=int, default=4, help="measurement repeats for averaging, 4 by default")
    parser.add_argument("-l", "--library_function", action="store_true", help="use scikit-learn's AgglomerativeClustering as the strict implementation instead of simple O(n^3) algorithm, measure larger data")
    parser.add_argument("-t", "--save_times", action="store_true", help="save times to .csv instead of printing")
    parser.add_argument("-p", "--save_plot", action="store_true", help="save plot to .pdf instead of showing")
    parser.add_argument("-f", "--filename", choices=names_roots, default="corners", help="specific file to measure, 'corners' by default")
    parser.add_argument("-o", "--output", default="results/time", help="output folder of result .csv files, results/time by default")
    parser.add_argument("-i", "--image_output", default="results/time",
                        help="output folder of result plots, results/time by default")
    args = parser.parse_args()
    resultsFolderPath = args.output
    imageFolderPath = args.image_output
    for folder in os.scandir("data"):
        for file in os.scandir(folder.path):
            full = np.genfromtxt(file.path, delimiter='\t')
            data = full[:, :-1]
            filename = Path(file.path).stem
            names.append(filename)
            fullData[filename] = DataEntry(data, None, filename, dim=np.shape(data)[1])
            for key, value in num_of_clusters.items():
                if key in filename:
                    fullData[filename].clusters_number = value
                    break
    compare_time_complexity(resultsFolderPath, imageFolderPath, args.repeats, args.save_times, args.save_plot, args.library_function, args.filename)


if __name__ == "__main__":
    main()
