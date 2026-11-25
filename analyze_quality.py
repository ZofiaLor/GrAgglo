import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

names_roots = ["blobs", "circles", "corners", "crescents", "laguna", "spheres"]
granules_numbers = [50, 100, 200]
relation_types = ['t', 'e', 'g']
line_type = [':', '-', '--']
cmap = mpl.colormaps['tab10']
colors = cmap(np.linspace(0, 1, 9))
legend_labels = []
data_shapes = dict()
data_shapes_noise = dict()


def read_data(folderPath, shape_dep, avg, linkage, strict):
    shape_dep_text = "_shape_dep" if shape_dep else ""
    avg_text = "_avg" if avg else ""
    suffix = "_" + linkage + shape_dep_text + avg_text + "_accuracy.csv"
    for name in names_roots:
        filename = folderPath + "/" + name + suffix
        if not os.path.exists(filename):
            print(filename + " not found")
            return None, None
        data_shapes[name] = pandas.read_csv(filename)
        if shape_dep:
            df = data_shapes[name].copy()
            for index, row in df.iterrows():
                df.at[index, 'accuracy'] = row['accuracy'] * (1 - row['noise percentage'])
                df.at[index, 'recall'] = row['recall'] * (1 - row['noise percentage'])
                df.at[index, 'precision'] = row['precision'] * (1 - row['noise percentage'])
            data_shapes_noise[name] = df
        if strict:
            data_strict = pandas.read_csv(folderPath + "/" + name + "_" + linkage + "_strict_accuracy.csv")
            print(name, " mean for strict algorithm:\n", data_strict[["granules", "accuracy", "recall", "precision"]].groupby(by=["granules"]).mean())
            print(name, " standard deviation for strict algorithm:\n",
                  data_strict[["granules", "accuracy", "recall", "precision"]].groupby(by=["granules"]).std())

    if shape_dep:
        return pandas.concat(data_shapes.values()), pandas.concat(data_shapes_noise.values())
    return pandas.concat(data_shapes.values()), None


def mean_max_std(name, data):
    print("Mean - " + name)
    print(data.mean().to_string())
    print("\nMax - " + name)
    print(data.max().to_string())
    print("\nStandard deviation - " + name)
    print(data.std().to_string())


def analyze_data_shape(shape_dep):
    print("Data shape impact analysis")
    for name in names_roots:
        mean_max_std(name, data_shapes[name][["data size", "accuracy", "recall", "precision"]].groupby(by=["data size"]))
        if shape_dep:
            print("\nWith noise")
            mean_max_std(name, data_shapes_noise[name][["data size", "accuracy", "recall", "precision"]].groupby(by=["data size"]))


def analyze_relation_type(data, data_noise, shape_dep):
    print("Relation type impact analysis")
    print("Mean - all data")
    print(data[["relation type", "accuracy", "recall", "precision"]].groupby(by=["relation type"]).mean().to_string())
    if shape_dep:
        print("\nWith noise")
        print(data_noise[["relation type", "accuracy", "recall", "precision"]].groupby(
            by=["relation type"]).mean().to_string())
    for name in names_roots:
        mean_max_std(name, data_shapes[name][["relation type", "accuracy", "recall", "precision"]].groupby(by=["relation type"]))

        if shape_dep:
            print("\nWith noise")
            mean_max_std(name, data_shapes_noise[name][["relation type", "accuracy", "recall", "precision"]].groupby(
                by=["relation type"]))


def analyze_granule_number(data, data_noise, shape_dep):
    print("Granule number impact analysis")
    print("Mean - all data")
    print(data[["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]).mean().to_string())
    if shape_dep:
        print("\nWith noise")
        print(data_noise[["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]).mean().to_string())
    for name in names_roots:
        mean_max_std(name, data_shapes[name][["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]))

        if shape_dep:
            print("\nWith noise")
            mean_max_std(name, data_shapes_noise[name][["granules number", "accuracy", "recall", "precision"]].groupby(by=["granules number"]))


def analyze_xi(data, data_noise, shape_dep):
    print("Xi impact analysis")
    print("Mean")
    to_group = data[["xi", "accuracy", "recall", "precision"]].groupby(by=["xi"], as_index=False)
    grouped = to_group.mean()
    print(grouped.to_string())
    plt.figure(figsize=(9,9))
    plt.plot(grouped["xi"], grouped["accuracy"], marker='o')
    plt.plot(grouped["xi"], grouped["recall"], marker='o')
    plt.plot(grouped["xi"], grouped["precision"], marker='o')
    plt.legend(["accuracy", "recall", "precision"])
    plt.title(r'Average accuracy, recall and precision depending on the $ \xi $ parameter')
    plt.xlabel(r'$ \xi $')
    plt.ylabel('result')
    plt.show()

    if shape_dep:
        print("\nWith noise")
        print("mean")
        to_group = data_noise[["xi", "accuracy", "recall", "precision"]]
        grouped = to_group.groupby(by=["xi"], as_index=False).mean()
        print(grouped.to_string())
        plt.figure(figsize=(9, 9))
        plt.plot(grouped["xi"], grouped["accuracy"], marker='o')
        plt.plot(grouped["xi"], grouped["recall"], marker='o')
        plt.plot(grouped["xi"], grouped["precision"], marker='o')
        plt.legend(["accuracy", "recall", "precision"])
        plt.title(r'Average accuracy, recall and precision depending on the $ \xi $ parameter, with noise')
        plt.xlabel(r'$ \xi $')
        plt.ylabel('result')
        plt.show()


def analyze_combination(folder, linkage):
    metric_names = ["accuracy", "recall", "precision"]
    for name, shape in data_shapes.items():
        folderPath = folder + "/" + name + "/"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filtered_size = shape[["granules number", "relation type", "xi", "accuracy", "recall", "precision"]].groupby(by=["granules number", "relation type", "xi"]).mean().reset_index()

        for metric in metric_names:
            plt.figure(figsize=(12, 10))
            for i in range(3):
                for j in range(3):
                    d = filtered_size[(filtered_size['granules number'] == granules_numbers[i]) & (filtered_size['relation type'] == relation_types[j])]
                    plt.plot(d['xi'], d[metric], linestyle=line_type[i])
                    plt.title("Dependency of " + name + " data clustering " + metric + r" from granule number, relation type and $ \xi $ value with " + linkage + " linkage")
                    plt.xlabel(r"$ \xi $")
                    plt.ylabel(metric)
                    legend_labels.append(str(granules_numbers[i]) + " granules, relation type " + relation_types[j])
            plt.legend(legend_labels)
            # plt.show()
            plt.savefig(folderPath + name + "_complex_ " + metric + "_" + linkage + ".pdf")
            plt.close()


def main():
    parser = argparse.ArgumentParser(prog="Clustering Quality Analysis",
                                     description="Analyze quality results based on different parameters")
    parser.add_argument("-i", "--input", default="results/quality",
                        help="input folder with measurements results, results/quality by default")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("-s", "--shape_dependent", action="store_true", help="analyze results of when granule memberships are based on granule shape/relation type")
    parser.add_argument("-a", "--averages", action="store_true",
                        help="analyze results of measurements with repeats (with 'avg' in name)")
    parser.add_argument("-d", "--data_shape", action="store_true", help="analyze quality based on data shape")
    parser.add_argument("-r", "--relation_type", action="store_true", help="analyze quality based on relation type")
    parser.add_argument("-g", "--granule_number", action="store_true", help="analyze quality based on granule number")
    parser.add_argument("-x", "--xi", action="store_true", help="analyze quality based on xi (threshold) parameter")
    parser.add_argument("-c", "--complex", action="store_true", help="analyze quality based on the combination of relation type, granule number and xi parameter")
    parser.add_argument("-o", "--output", default="results/img/complex", help="output folder of complex analysis plots, results/img/complex by default")

    parser.add_argument("--strict", action="store_true",
                        help="analyze strict (non-fuzzy) clustering results in addition to fuzzy ones")
    args = parser.parse_args()

    data, data_noise = read_data(args.input, args.shape_dependent, args.averages, args.linkage, args.strict)
    if data is None:
        return
    if args.data_shape:
        analyze_data_shape(args.shape_dependent)
    if args.relation_type:
        analyze_relation_type(data, data_noise, args.shape_dependent)
    if args.granule_number:
        analyze_granule_number(data, data_noise, args.shape_dependent)
    if args.xi:
        analyze_xi(data, data_noise, args.shape_dependent)
    if args.complex:
        analyze_combination(args.output, args.linkage)


if __name__ == "__main__":
    main()
