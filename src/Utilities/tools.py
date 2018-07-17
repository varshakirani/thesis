from argparse import ArgumentParser

import json
import scipy.io as sio

import sys
import os
import pandas as pd


def parse_options():
    parser = ArgumentParser()
    #parser.add_argument("-a", "--all", required=False, default=False,
    #                   action="store_true",
    #                   help="Run all the ML algorithms.")

    parser.add_argument("-n", "--number_iterations", required=False,
                        default=100, type=int,
                        help="Number of iterations to run the cross validation")

    parser.add_argument("-k", "--kFold", required=False,
                        default=10, type=int,
                        help="k fold number in Stratified Cross Validation")

    parser.add_argument( "--nClass", required=False,
                        default=3, type=int,
                        help="3 class or 2 class classifier.")

    parser.add_argument("-d", "--data", required=False,
                        default="../../Data", type=str,
                        help="Path to data folder")

    parser.add_argument("-m", "--model", required=False,
                        default="all", type=str,
                        help="Model name to run. pass 'all' to run all the models")

    parser.add_argument("-o", "--output", required=False,
                        default="outputs/", type=str,
                        help="Output Folder where results are to stored")

    options = parser.parse_args()
    return options

def data_extraction(data_folder, nClass):
    """
    :param data_folder: Path to the folder that contains Data
    :return: df: When nClass=3 Single panda dataframe containing means of various Region of interest (ROI) of Brain of all the three classes combined
            df1, df2, df3: Separated dataframes for each class when nClass is 2
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

    if nClass == 3: #No need for separated data
        return df

    elif nClass == 2:
        df1 = df[df.label == 1]
        df2 = df[df.label == 2]
        df3 = df[df.label == 3]
        return df1, df2, df3

def dump_results_to_json(model_name, results, output_folder):
    """
    :param model_name: Machine learning model name
    :param results: Scores of kFold stratified cross Validation
    :return:
    """

    res_file = open(output_folder+"results_%s.json" % model_name, "w", encoding='utf-8' )
    #jsonList = [o.__dict__ for o in results]


    jsonList = [o.tolist() for o in results]
    json.dumps(jsonList)

    json.dump(jsonList, res_file, sort_keys=True, indent=4)



