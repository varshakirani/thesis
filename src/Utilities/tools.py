from argparse import ArgumentParser

import json
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

    parser.add_argument("-m", "--model", required=False,
                        default="svm", type=str,
                        help="Model name to run. pass 'all' to run all the models")

    options = parser.parse_args()
    #if not (options.all or options.svm or options.naive_bayes):
     #   print("You need to set all or svm or naive_bayes flag" )
      #  sys.exit(1)
    return options

def data_extraction(data_folder):
    """
    :param data_folder: Path to the folder that contains Data
    :return: panda dataframe containing means of various Region of interest (ROI) of Brain
    """

    data = sio.loadmat(data_folder+"/Faces_con_0001.mat")

    columns = ["means", "label"]
    # df = pd.DataFrame({'A':1,'B':2}, index = None)
    data_list = []
    for matFile in os.listdir(data_folder):
        if matFile.startswith("Faces") and not matFile.endswith("389.mat"):
            data = sio.loadmat(data_folder +"/" + matFile)
            for i in range(len(data["means"])):
                d = data["means"][i], data["label"][0][i]
                data_list.append(d)

    df = pd.DataFrame(data_list, columns=columns)
    RoiNames = (data["RoiName"][:, 0])
    colRoi = []
    for roi in RoiNames:
        colRoi.append(roi[0])
    df[colRoi] = pd.DataFrame(df.means.values.tolist(), index=df.index)
    df.drop(['means'], axis=1, inplace=True)

    return df

def dump_results_to_json(model_name, results):
    """
    :param model_name: Machine learning model name
    :param results: Scores of kFold stratified cross Validation
    :return:
    """

    res_file = open("results_%s.json" % model_name, "w", encoding='utf-8' )
    #jsonList = [o.__dict__ for o in results]


    jsonList = [o.tolist() for o in results]
    json.dumps(jsonList)

    json.dump(jsonList, res_file, sort_keys=True, indent=4)



