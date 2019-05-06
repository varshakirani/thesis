import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def get_avg_original_accuracy(raw_folder):
    original_files = list( filter(lambda x: x.startswith('permutation_result_'), os.listdir(raw_folder)))
    result = pd.DataFrame(columns=['contrast', 'class', 'Model', 'Avg original accuracy'])
    for file in original_files:
        df = pd.read_csv(raw_folder+file)
        res = pd.DataFrame({"Avg original accuracy":
                        df.groupby(['contrast','class','Model'])['original_accuracy'].mean()}).reset_index()
        result = pd.concat([res,result], ignore_index=True)
    return result

def calculate_pvalue(raw_folder, df_orig):
    per_files = list(filter(lambda x: x.startswith('n') or x.startswith('F'), os.listdir(raw_folder)))
    contrast_names = {
        'n2' : 'nBack_con_0002',
        'n3' : 'nBack_con_0003',
        'F3' : 'Faces_con_0003',
        'F4' : 'Faces_con_0004',
        'F5' : 'Faces_con_0005'
    }
    df_pvalues = pd.DataFrame(columns=['contrast', 'Model', 'class', "Avg original accuracy", "p_value"])
    for file in per_files:
        contrast = contrast_names[file[0:2]]
        classifier = file[3:5]
        model = file[6:].split(".")[0]

        original_accuracy = df_orig[(df_orig['contrast'] == contrast) & (df_orig['Model'] == model)
                                        & (df_orig['class'] == int(classifier) )]["Avg original accuracy"].values[0]

        df_perm = pd.read_csv(raw_folder+file, header=None, names=['performance_scores'])

        #performance_scores = np.asarray(df_perm['performance_scores'])
        #p_value = sum(performance_scores >= original_accuracy)/10000
        p_value = df_perm[df_perm['performance_scores'] >= original_accuracy]['performance_scores'].count()/10000
        df_pvalues = df_pvalues.append({'contrast': contrast,
                                        "Model":model,
                                        "class":int(classifier),
                                        "Avg original accuracy":original_accuracy,
                                       "p_value":p_value},
                                       ignore_index=True)
    return df_pvalues

if __name__ == "__main__":
    raw_folder = "Ultimate_final_output/no_gender/permutation/"

    df = get_avg_original_accuracy(raw_folder)
    df_pvalues = calculate_pvalue(raw_folder, df)
    print(df_pvalues)