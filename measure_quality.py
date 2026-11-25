import os
import pandas
import numpy as np
from pathlib import Path
from data_entry import DataEntry
import argparse

# Constants
# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}
names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
data_size_presets = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]

fullData = {}
names = []


def measure_metrics_repeats(folderPath, linkage, repeats, specific_file=None, shape_dependent=False):
    suffix = "_shape_dep_avg_accuracy.csv" if shape_dependent else "_avg_accuracy.csv"
    if specific_file is not None:
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            filepath = folderPath + "/" + specific_file + "_" + linkage + suffix
            if not os.path.exists(filepath):
                os.makedirs(folderPath, exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write("data size,granules number,relation type,xi,accuracy,recall,precision,noise percentage")
            data_to_measure.measure_accuracy_repeats(filepath, shape_dependent, linkage, repeats)
    else:
        for root in names_roots:
            filepath = folderPath + "/" + root + "_" + linkage + suffix
            if not os.path.exists(filepath):
                os.makedirs(folderPath, exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write("data size,granules number,relation type,xi,accuracy,recall,precision,noise percentage")
            for data_size in data_size_presets:
                print(root + str(data_size))
                fullData[root + str(data_size)].measure_accuracy_repeats(
                    filepath, shape_dependent_membership=shape_dependent, linkage=linkage, repeats=repeats)


def measure_metrics_const_seed(folderPath, linkage, seed, specific_file=None, shape_dependent=False):
    if specific_file is not None:
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            print(data_to_measure.measure_accuracy_const_seed(seed, shape_dependent, linkage).to_string())
    else:
        for root in names_roots:
            accuracy_results = pandas.DataFrame()
            for data_size in data_size_presets:
                print(root + str(data_size))
                result = fullData[root + str(data_size)].measure_accuracy_const_seed(
                    seed, shape_dependent_membership=shape_dependent,linkage=linkage)
                accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            if shape_dependent:
                accuracy_results.to_csv(folderPath + root + "_" + linkage + "_shape_dep_accuracy.csv")
            else:
                accuracy_results.to_csv(folderPath + root + "_" + linkage + "_accuracy.csv")


def measure_strict_accuracy_recall_precision(folderPath, linkage, specific_file=None):
    linkages = {"single": "sl", "complete": "cl"}
    if specific_file is not None:
        data_to_measure = fullData.get(specific_file)
        if data_to_measure is not None:
            print(data_to_measure.measure_strict_accuracy(linkage))
    else:
        for root in names_roots:
            accuracy_results = pandas.DataFrame()
            for data_size in data_size_presets:
                print(root + str(data_size))
                result = fullData[root + str(data_size)].measure_strict_accuracy(linkage=linkage)
                accuracy_results = pandas.concat([accuracy_results, result], ignore_index=True)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            accuracy_results.to_csv(folderPath + root + "_" + linkages[linkage] + "_strict_accuracy.csv")


def main():
    parser = argparse.ArgumentParser(prog="Measure Quality",
                                     description="Measure quality (accuracy, recall, precision) of fuzzy hierarchical clusterization")
    parser.add_argument("-f", "--filename",
                        help="specific file to measure, measures all files if not given; result is printed for one file, saved to .csv for all files")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("--relation_dependent", action="store_true",
                        help="calculate granule memberships based on granule shape/relation type")
    parser.add_argument("--strict", action="store_true",
                        help="measure strict (non-fuzzy) clustering results in addition to fuzzy ones")
    parser.add_argument("-o", "--output", default="results/quality",
                        help="output folder of measurements, results/quality by default")
    parser.add_argument("-s", "--seed", type=int,
                        help="random generation seed for FCM, if set a faster evaluation algorithm is used")
    parser.add_argument("-r", "--repeats", type=int, default=10,
                        help="number of measurements for each parameter combination, used only when seed is None")
    args = parser.parse_args()
    folderPath = args.output
    for folder in os.scandir("data"):
        for file in os.scandir(folder.path):
            full = np.genfromtxt(file.path, delimiter='\t')
            data = full[:, :-1]
            l = full[:, -1:]
            labels = np.zeros(np.shape(l)[0], dtype=int)
            for i in range(np.shape(l)[0]):
                labels[i] = int(l[i][0])
            filename = Path(file.path).stem
            names.append(filename)
            fullData[filename] = DataEntry(data, labels, filename, dim=np.shape(data)[1])
            for key, value in num_of_clusters.items():
                if key in filename:
                    fullData[filename].clusters_number = value
                    break

    if args.seed is None:
        measure_metrics_repeats(folderPath, args.linkage, args.repeats, args.filename, args.relation_dependent)
    else:
        measure_metrics_const_seed(folderPath, args.linkage, args.seed, args.filename, args.relation_dependent)
    if args.strict:
        measure_strict_accuracy_recall_precision(folderPath, args.linkage, args.filename)


if __name__ == "__main__":
    main()
