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
    parser.add_argument("-t", "--type", required=False,
                        default=0, type=int,
                        help="0 for old files and 1 for new files ")
    parser.add_argument("-b",'--box_plot', required=False,
                        default=False,  action="store_true",
                        help="Box plot per contrast")
    parser.add_argument("-p", '--performance_plot', required=False,
                       default=False, action="store_true",
                       help="Performance of different Contrasts")

    options = parser.parse_args()
    return options


def plot_boxplot(rows, cols, title, x, y, scoresdf, options):
    models = scoresdf['Model'].unique()
    nClass = scoresdf['Classifier'].unique()
    contrasts = scoresdf['Contrast_name'].unique()
    for contrast in contrasts:
        rows_num = rows
        col_nums = cols
        fig, axes = plt.subplots(nrows=rows_num, ncols=col_nums, figsize=(20, 20))
        axs = axes.ravel()
        j = 0
        for nc in nClass:

            for model_name in models:
                ax = sns.boxplot(x=x, y=y, hue='Type',
                                 data=scoresdf[(scoresdf['Model'] == model_name) & (scoresdf['Classifier'] == nc)
                                               & (scoresdf['Contrast_name'] == contrast)], ax=axs[j])

                axs[j].set_xlabel('')
                axs[j].set_ylabel('')

                j = j + 1
            axs[j - cols].set_ylabel(str(nc))
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.suptitle(contrast+title)
        plt.savefig("%s%s_%s.png" % (options.output, contrast, title))
        plt.cla()
        plt.clf()
        plt.close()

def performance_plots(df):
    all_c = df['Contrast_name'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
    axs = axes.ravel()

    sns.pointplot(x='Model',y='Score',hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[0]).set_title('Faces 12')

    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[1]).set_title('Faces 23')
    sns.pointplot(x='Model',y='Score',hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[2]).set_title('Faces 31')
    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[3]).set_title('Faces 123')
    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[4]).set_title('nBack 12')
    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[5]).set_title('nBack 23')
    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[6]).set_title('nBack 31')
    sns.pointplot(x='Model', y='Score', hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[7]).set_title('nBack 123')
    for i in range(8):
        axs[i].set_xticklabels(['rbf_b', 'rbf', 'nb', 'dt', 'rfc', 'lr'])
        axs[i].set_xlabel('')
        #axs[i].set_ylim(0.25,0.69 )
        if i == 3 :
            continue
        elif i == 7:
            continue
        axs[i].legend_.remove()
    for i in range(4,8,1):
        axs[i].set_xlabel('Models')

    plt.show()

def main():
    options = parse_options()
    if options.type == 1:
        tun = options.input.split('/')[-1].split('.')[-2]
    else:
        tun = options.input.split('/')[-1].split('.')[-2].split('_')[-2]
    scoresdf = pd.read_csv(options.input)
    if options.box_plot:
        col = len(scoresdf['Model'].unique())
        title = tun+'_OverallAccuracy'
        plot_boxplot(4,col,title,'Model','Score',scoresdf,options)

        title = tun + '_BalancedAccuracy'
        plot_boxplot(4, col, title, 'Model', 'Balanced_accuracy', scoresdf, options)
        """
    
    models = scoresdf['Model'].unique()
    nClass = scoresdf['Classifier'].unique()
    contrasts = scoresdf['Contrast_name'].unique()
    for contrast in contrasts:
        rows_num = 4
        col_nums = col
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
        plt.suptitle(contrast)
        plt.savefig("%s%s_%s.png"%(options.output,contrast,tun))
        plt.cla()
        plt.clf()
        plt.close()
        """
    if options.performance_plot:
        performance_plots(scoresdf)


if __name__ == '__main__':
    main()