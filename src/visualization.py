import os
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from statistics import mean



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

    print("Visualization of Training scores")
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axs = axes.ravel()
    i = 0
    for c in classes:

        results_train = {}

        for json_file in os.listdir(options.output):
            data_type = json_file.split(".")[0].split("_")[-2]

            if json_file.split(".")[0].split("_")[-1] == c and data_type == "train":
                model_name = json_file.split(".")[0][8:]
                results_train[model_name] = json.load(open(options.output+"/"+json_file))
        for model, scores in results_train.items():
            scores = array(scores)
            mean_scores = scores.mean(axis=1)
            axs[i].hist(mean_scores, 100, label="%s. Mean:%s" % (model, mean_scores.mean(axis=0)))
            axs[i].legend(loc = "upper right")
            axs[i].set_title(c)

        i = i+1
    plt.suptitle("Training Scores")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.savefig("out/visualization/train_hist.png")
    plt.show()

    print("Visualization of testing scores")
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axs = axes.ravel()
    i = 0

    for c in classes:

        results_test = {}

        for json_file in os.listdir(options.output):
            data_type = json_file.split(".")[0].split("_")[-2]

            if json_file.split(".")[0].split("_")[-1] == c and data_type == "test":

                model_name = json_file.split(".")[0][8:]
                results_test[model_name] = json.load(open(options.output + "/" + json_file))

        for model, scores in results_test.items():
            scores = array(scores)
            axs[i].hist(scores, 100,  label="%s. Mean:%s" % (model, mean(scores)))
            axs[i].legend(loc="upper right")
            axs[i].set_title(c)

        i = i + 1
    plt.suptitle("Testing Scores")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.savefig("out/visualization/test_hist.png")
    plt.show()
    print("Visualization of both train and test ")
    fig, axes = plt.subplots(nrows=4, ncols=3)
    axs = axes.ravel()
    i = 0

    results_test = {}
    results_train = {}
    for json_file in os.listdir(options.output):
        data_type = json_file.split(".")[0].split("_")[-2]
        if data_type == "train":
            model_name = json_file.split(".")[0][8:]
            results_train[model_name] = json.load(open(options.output + "/" + json_file))

        elif data_type == "test":
            model_name = json_file.split(".")[0][8:]
            results_test[model_name] = json.load(open(options.output + "/" + json_file))

    for (model_train, scores_train), (model_test, scores_test) in zip(results_train.items(), results_test.items()):
        class_det = model_train.split("_")[-1]
        model_name = model_train.split("_")[0:-2]
        model_name = " ".join(model_name)
        model_name = model_name + " " + class_det
        axs[i].plot(array(scores_train).mean(axis=1), color='green', alpha=0.8, label='Train %s'%(mean(array(scores_train).mean(axis=1))))
        axs[i].plot(array(scores_test), color='magenta', alpha=0.8, label='Test %s'%(mean(scores_test)))
        axs[i].set_title("%s" % (model_name), fontsize=14)
        # axs[i].xlabel('Epochs')
        axs[i].legend(loc='upper left')
        i = i + 1

    plt.suptitle("Both Training and testing")

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.savefig("out/visualization/train_test.png")
    plt.show()


####################
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axs = axes.ravel()
    i = 0

    for c in classes:

        results_test = {}

        for json_file in os.listdir(options.output):
            data_type = json_file.split(".")[0].split("_")[-2]

            if json_file.split(".")[0].split("_")[-1] == c and data_type == "test":

                model_name = json_file.split(".")[0][8:]
                results_test[model_name] = json.load(open(options.output + "/" + json_file))

        for model, scores in results_test.items():
            scores = np.array(scores)
            x_values = np.arange(len(scores))
            label_str = "%s. Mean:%s" % (' '.join(map(str, model.split("_")[:-2])), mean(scores))
            axs[i].plot(x_values, scores, label=label_str)
            axs[i].legend(loc="upper right")
            axs[i].set_title(c)
            axs[i].set_ylim([0, 1])

        i = i + 1
    plt.suptitle("Testing Scores")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.savefig("out/visualization/test_line.png")
    plt.show()


if __name__ == "__main__":
    main()