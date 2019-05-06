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

if __name__ == "__main__":
    raw_folder = "Ultimate_final_output/no_gender/permutation/"

    df = get_avg_original_accuracy(raw_folder)
    print(df)