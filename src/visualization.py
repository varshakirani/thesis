import os
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import array



def parse_options():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", required=False,
                        default="outputs", type=str,
                        help="Path to output folder")

    options = parser.parse_args()
    return options


def main():
    print("Visualization of cross validation score")
    options = parse_options()
    classes = ["12", "23", "31", "123"]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axs = axes.ravel()
    i = 0
    for c in classes:

        results = {}

        for json_file in os.listdir(options.output):
            if json_file.split(".")[0].split("_")[-1] == c:

                model_name = json_file.split(".")[0][8:]
                results[model_name] = json.load(open(options.output+"/"+json_file))

        for model, scores in results.items():
            scores = array(scores)
            mean_scores = scores.mean(axis=1)
            axs[i].hist(mean_scores, 100, label="%s. Mean:%s" % (model, mean_scores.mean(axis=0)))
            axs[i].legend(loc = "upper right")
            axs[i].set_title(c)

        i = i+1

    plt.show()
    #plt.savefig("mv1")

if __name__ == "__main__":
    main()