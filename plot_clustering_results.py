import os
import numpy as np
from pathlib import Path
import argparse
from data_entry import DataEntry

# Constants
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}

fullData = {}
names = []


def plot_results(folderPath, specific_name=None, save_plot=True, linkage="single", grayscale=False):
    if save_plot and not os.path.exists(folderPath):
        os.makedirs(folderPath)
    if specific_name is None:
        for name in names:
            for r in ['t', 'e', 'g']:
                filename = ''.join([folderPath, '/', name, r, '_', linkage[0], '.pdf'])
                fullData[name].fit_plot_fuzzy_labels(r, linkage=linkage, file_path=filename, grayscale=grayscale)
    else:
        data_to_plot = fullData.get(specific_name)
        if data_to_plot is not None:
            filename = None
            for r in ['t', 'e', 'g']:
                if save_plot:
                    filename = ''.join([folderPath, '/', specific_name, r, '_', linkage[0], '.pdf'])
                data_to_plot.fit_plot_fuzzy_labels(r, linkage=linkage, file_path=filename, grayscale=grayscale)


def main():
    parser = argparse.ArgumentParser(prog="Plot Granule Clusterization",
                                     description="Plot results of fuzzy hierarchical clusterization for 3 relation types, granule numbers and xi values")
    parser.add_argument("-f", "--filename", help="specific file to plot results of, all files if not given")
    parser.add_argument("-g", "--grayscale", action="store_true", help="plots in grayscale")
    parser.add_argument("-l", "--linkage", choices=["single", "complete"], default="single",
                        help="linkage of the hierachical clustering: single or complete, single by default")
    parser.add_argument("-p", "--save_plot", action="store_true", help="save plot to .pdf instead of showing (always saves for all files plotting)")
    parser.add_argument("-o", "--output", default="results/img", help="output folder of plots, results/img by default")
    args = parser.parse_args()
    folderPath = args.output
    for folder in os.scandir("data"):
        for file in os.scandir(folder.path):
            full = np.genfromtxt(file.path, delimiter='\t')
            data = full[:, :-1]
            labels = full[:, -1:]
            filename = Path(file.path).stem
            names.append(filename)
            if "spheres" in file.name:
                fullData[file.name[:-5]] = DataEntry(data, labels, filename, 3)
            else:
                fullData[file.name[:-5]] = DataEntry(data, labels, filename)
            for key, value in num_of_clusters.items():
                if key in filename:
                    fullData[filename].clusters_number = value
                    break
    plot_results(folderPath, args.filename, args.save_plot, args.linkage, args.grayscale)


if __name__ == "__main__":
    main()
