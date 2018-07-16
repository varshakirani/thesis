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
    results = {}
    for json_file in os.listdir(options.output):
        model_name = json_file.split(".")[0][8:]
        results[model_name] = json.load(open(options.output+"/"+json_file))

    for model, scores in results.items():
        scores = array(scores)
        mean_scores = scores.mean(axis=1)
        plt.hist(mean_scores, 100,  label="%s. Mean:%s"%(model,mean_scores.mean(axis=0)))
    ylim = plt.ylim()
        #plt.plot(2 * [mean_scores.mean(axis=0)], ylim, "--g", linewidth=3, label="Mean Score of 100 iterations. Value=%s"%(mean_scores.mean(axis=0)))
    plt.xlabel("Mean Scores")
    plt.legend(loc = "upper right")

    plt.ylim(ylim)

    plt.show()

if __name__ == "__main__":
    main()