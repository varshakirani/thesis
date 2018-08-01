import os
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from statistics import mean
import pandas as pd
import seaborn as sns



def parse_options():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", required=False,
                        default="outputs", type=str,
                        help="Path to output folder")

    options = parser.parse_args()
    return options

def main():
    options = parse_options()
    scoresdf = pd.read_csv(options.output+'without_kfold.csv')
    models = scoresdf['Model'].unique()
    nClass = scoresdf['Classifier'].unique()
    contrasts = scoresdf['Contrast_name'].unique()
    for contrast in contrasts:
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
        axs = axes.ravel()
        j = 0
        for nc in nClass:
            for model_name in models:

                #sns.boxplot(x='Model', y='Score',
                 #           data=scoresdf[(scoresdf['Type'] == 'test') & (scoresdf['Model'] == model_name)], ax=axs[j])
                ax = sns.boxplot(x='Model',y='Score', hue='Type',data=scoresdf[(scoresdf['Model'] == model_name) & (scoresdf['Classifier'] == nc)], ax=axs[j])

                axs[j].set_xlabel('')
                axs[j].set_ylabel('')

                j= j+1

            axs[j-3].set_ylabel(str(nc))

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        #print(*mng.window.maxsize())

        #plt.figure(figsize=(1440, 797))

        plt.suptitle(contrast)
        plt.savefig("out/outputs_new/%s_withoutkfold.png"%(contrast))
        plt.cla()
        plt.clf()
        plt.close()
        #plt.show()


if __name__ == '__main__':
    main()