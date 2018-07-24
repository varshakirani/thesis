import matplotlib.pyplot as plt
import sys
import seaborn as sns
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')

import tools
import ml_utilities as mlu
if __name__ == "__main__":

    df1 = tools.data_extraction("../Data",3, "Faces_con_0001.mat" )
    df2 = tools.data_extraction("../Data",3,"Faces_con_0002.mat")

    df1 = mlu.missing_values(df1, 1)
    roi_len = len(df1.columns) - 2  # not considering label and subject_cont
    roi_len = 4
    for j in range(int(roi_len / 4)):
        plt.figure()
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axs = axes.ravel()

        for i in range(4):
            f = df1.iloc[:, (4 * j) + i + 1]
            p = sns.distplot(f, ax=axs[i])
            p.set_title(f.name)
            p.set_xlabel(" ")

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.savefig("out/data_explores/allclass_%s.png"%(j))




