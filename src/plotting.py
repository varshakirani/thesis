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

    parser.add_argument('-i','--input', required=True,
                        default='out/output_scores_testing/Faces_con_0001&Faces_con_0001_389.csv', type=str,
                        help='Path to input csv file which contains information about the scores ')
    options = parser.parse_args()
    return options

def main():
    options = parse_options()

    tun = options.input.split('/')[-1].split('.')[-2].split('_')[-2]
    #tun = 'without_tuning'
    print(tun)
    scoresdf = pd.read_csv(options.input)
    print(scoresdf.shape)
    models = scoresdf['Model'].unique()
    nClass = scoresdf['Classifier'].unique()
    contrasts = scoresdf['Contrast_name'].unique()
    for contrast in contrasts:
        rows_num = 4
        col_nums = 4
        fig, axes = plt.subplots(nrows=rows_num, ncols=col_nums, figsize=(20, 20))
        axs = axes.ravel()
        j = 0
        for nc in nClass:
            axs[j - 3].set_ylabel(str(nc))
            for model_name in models:

                ax = sns.boxplot(x='Model',y='Score', hue='Type',data=scoresdf[(scoresdf['Model'] == model_name) & (scoresdf['Classifier'] == nc)], ax=axs[j])

                axs[j].set_xlabel('')
                axs[j].set_ylabel('')

                j= j+1



        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        #print(*mng.window.maxsize())

        #plt.figure(figsize=(1440, 797))

        plt.suptitle(contrast)
        plt.savefig("%s%s_%s.png"%(options.output,contrast,tun))
        plt.cla()
        plt.clf()
        plt.close()
        #plt.show()


if __name__ == '__main__':
    main()