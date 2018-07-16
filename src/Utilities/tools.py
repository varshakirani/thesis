from argparse import ArgumentParser

import scipy.io as sio

import sys
import os
import pandas as pd

def parse_options():
    parser = ArgumentParser()
    parser.add_argument("-a", "--all", required=False, default=False,
                        action="store_true",
                        help="Run all the ML algorithms.")

    parser.add_argument("-s", "--svm", required=False, default=False,
                        action="store_true",
                        help="Run SVM ML algorithm.")

    parser.add_argument("-nb", "--naive_bayes", required=False, default=False,
                        action="store_true",
                        help="Run naive-bayes ML algorithm.")

    parser.add_argument("-n", "--number_iterations", required=False,
                        default=100, type=int,
                        help="Number of iterations to run the cross validation")

    parser.add_argument("-k", "--kFold", required=False,
                        default=10, type=int,
                        help="k fold number in Stratified Cross Validation")

    parser.add_argument("-d", "--data", required=False,
                        default="../../Data", type=str,
                        help="Path to data folder")
    options = parser.parse_args()
    if not (options.all or options.svm or options.naive_bayes):
        print("You need to set all or svm or naive_bayes flag" )
        sys.exit(1)
    return options

def data_extraction(data_folder):
    """
    :param data_folder: Path to the folder that contains Data
    :return: panda dataframe containing means of various Region of interest (ROI) of Brain
    """
    print("inside data_extraction function")
    data = sio.loadmat(data_folder+"/Faces_con_0001.mat")
    print(data_folder)

    columns = ["means", "label"]
    # df = pd.DataFrame({'A':1,'B':2}, index = None)
    data_list = []
    for matFile in os.listdir(data_folder):
        if matFile.startswith("Faces") and not matFile.endswith("389.mat"):
            data = sio.loadmat(data_folder +"/" + matFile)
            for i in range(len(data["means"])):
                d = data["means"][i], data["label"][0][i]
                data_list.append(d)
            print(len(data["means"]))
    df = pd.DataFrame(data_list, columns=columns)
    RoiNames = (data["RoiName"][:, 0])
    colRoi = []
    for roi in RoiNames:
        colRoi.append(roi[0])
    df[colRoi] = pd.DataFrame(df.means.values.tolist(), index=df.index)
    df.drop(['means'], axis=1, inplace=True)

    return df
