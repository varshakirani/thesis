import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
import seaborn as sns
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')

import tools
import ml_utilities as mlu

def parse_options():
    parser = ArgumentParser()
    #parser.add_argument("-a", "--all", required=False, default=False,
    #                   action="store_true",
    #                   help="Run all the ML algorithms.")

    parser.add_argument("-u","--univariate", required=False, default=False, action="store_true", help="This option enables "
                                                                                              "univariate analysis of "
                                                                                              "all the 116 features"
                                                                                              "among different classes")

    parser.add_argument("-c","--correlate", required=False, default=False, action="store_true", help="This option enables "
                                                                                                 "correlation of first 16 "
                                                                                                "features among all classes")

    parser.add_argument("--heatmap", required=False, default=False, action="store_true", help="This option enables "
                                                                                                   "correlation of all "
                                                                                                    "features among all classes and plots heatmap")

    parser.add_argument("-m", "--missing_data_plot", required=False, default=False, action="store_true",
                        help="This option plots missing data on a heat map")
    options = parser.parse_args()
    return options

def univariate_analysis(df1):
    roi_len = len(df1.columns) - 2  # not considering label and subject_cont
    labels = [1, 2, 3]
    sns.set_context("paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 5})
    for j in range(int(roi_len / 4)):
        plt.figure()
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axs = axes.ravel()

        for i in range(4):
            for l in labels:
                f = df1[df1["label"] == l].iloc[:, (4 * j) + i + 1]
                if l == 1:
                    legend_str = 'Bipolar'
                elif l == 2:
                    legend_str = 'Schizophrenia'
                elif l == 3:
                    legend_str = 'Control'
                #legend_str = "class" + str(l)
                p = sns.distplot(f, ax=axs[i], label=legend_str)
                p.set_title(f.name)
                p.set_xlabel(" ")
                p.legend()

        #for i in range(4):
        #    f = df1.iloc[:, (4 * j) + i + 1]
        #    legend_str = "all_class"
        #    p = sns.distplot(f, ax=axs[i], label=legend_str)
        #    p.set_title(f.name)
        #    p.set_xlabel(" ")
        #    p.legend()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.savefig("out/data_exploration/distribution_plots/class_%s.png" % (j))

def features_correlation(df1):
    ### This section is to find the correlation between the attributes

    roi = df1[df1.columns[1:116]].columns

    for i in range(0, 16, 4):

        if i + 4 < 16:
            sns.pairplot(df1, kind="scatter", hue="label", markers=["o", "s", "D"], vars=roi[i:i + 4], palette="Set2")
            plt.savefig("out/data_exploration/correlation_plots/correl_%s.png" % (i))


def corr_heatmap(df1,title):
    corr = df1.corr()
    print(corr.columns)
    # plot the heatmap
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns).set_title(title)

    plt.savefig("out/data_exploration/correlation_plots/heat_map_%s"%(title) )
    plt.show()


def missdata_plot(df1, title):
    sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis').set_title(title)
    plt.savefig("out/data_exploration/missing_data.png")
    plt.show()

if __name__ == "__main__":

    df1,c = tools.data_extraction("../Data",3, "Faces_con_0001.mat" )
    df2,c = tools.data_extraction("../Data",3,"Faces_con_0002.mat")

    #df1 = mlu.missing_values(df1, 1)

    options = parse_options()
    if options.univariate:
        univariate_analysis(df1)

    if options.correlate:
        features_correlation(df1)

    if options.heatmap:
        corr_heatmap(df1[df1["label"] == 1], "Bipolar Disorder Subjects")
        corr_heatmap(df1[df1["label"] == 2], "Schizophrenia Subjects")
        corr_heatmap(df1[df1["label"] == 3], "Control Subjects")

    if options.missing_data_plot:
        missdata_plot(df1,"Faces_con_0001" )






