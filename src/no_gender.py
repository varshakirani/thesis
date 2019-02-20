import sys
sys.path.insert(0, 'src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os
import time
import re
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    """
    :param df: data frame which contains mean activation values in 116 brainn areas
    :return: male and female combined training and testing data where testing data is standardized using mean and variance of
             the respective gender training data.
    """

    # Split the data into 80% training and 20% testing.
    train, test = mlu.train_test_split(df)

    # Obtaining male and female dataframes
    train_male = train.loc[train["gender"] == 1]
    train_female = train.loc[train["gender"] == 2]
    test_male = test.loc[test["gender"] == 1]
    test_female = test.loc[test["gender"] == 2]

    # Removing age and gender info from dataframe, so that only mean activation values in 116 brain regions are considered.
    train_male = train_male.drop(['gender', 'age'], axis=1, errors='ignore')
    train_female = train_female.drop(['gender', 'age'], axis=1, errors='ignore')
    test_male = test_male.drop(['gender', 'age'], axis=1, errors='ignore')
    test_female = test_female.drop(['gender', 'age'], axis=1, errors='ignore')

    # Converting dataframes into X and Y arrays wrt male and female
    x_train_male, y_train_male = mlu.get_features_labels(train_male)
    x_train_female, y_train_female = mlu.get_features_labels(train_female)

    x_test_male, y_test_male = mlu.get_features_labels(test_male)
    x_test_female, y_test_female = mlu.get_features_labels(test_female)

    # Standardisation of male training data and female training data
    scaler_male = StandardScaler()
    scaler_female = StandardScaler()
    x_train_male = scaler_male.fit_transform(x_train_male, y_train_male)
    x_train_female = scaler_female.fit_transform(x_train_female, y_train_female)

    # Standardisation of male testing data using mean and variance from male training data scale.
    x_test_male = scaler_male.transform(x_test_male)

    # Standardisation of female testing data using mean and variance from female training data scale.
    x_test_female = scaler_female.transform(x_test_female)

    # Combining male training data and female training data.
    x_train = np.concatenate((x_train_male, x_train_female))
    y_train = np.concatenate((y_train_male, y_train_female))

    # Combining male testing data and female testing data.
    x_test = np.concatenate((x_test_male, x_test_female))
    y_test = np.concatenate((y_test_male, y_test_female))

    return x_train, y_train, x_test, y_test


def run_no_gender_ml(df, options, n, scoresdf, contrast_name):
    """
    :param df: Dataframe containing mean activation values in 116 brain areas
    :param options: info passed via command argument which contains details like file paths, etc
    :param n: represents classes considered. 123- for all, 12- Bipolar&schizo, 23- Schizo&Control, 31- Control&Bipolar
    :param scoresdf: Results dataframe containing scores
    :param contrast_name: Contrast Name
    :return: Results dataframe containing scores after elimination of gender effect on the data
    """
    print(contrast_name)
    models = ["svm_kernel_default", "svm_kernel_tuned", "naive_bayes", "decision_tree", "rfc", 'logistic_regression']
    for i in range(options.number_iterations):
        x_train, y_train, x_test, y_test = preprocess(df)

        for model_name in models:
            train_score, train_balanced_score, trained_model, min_max_scaler = mlu.model_fitting(model_name, x_train,
                                                                                                 y_train, 10, False)

            test_score = trained_model.score(x_test, y_test)
            test_balanced_score = mlu.balanced_accuracy(trained_model.predict(x_test), y_test)

            scoresdf = scoresdf.append(
                {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                 'Contrast_name': contrast_name, 'Balanced_accuracy': train_balanced_score}, ignore_index=True)
            scoresdf = scoresdf.append(
                {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                 'Contrast_name': contrast_name, 'Balanced_accuracy': test_balanced_score}, ignore_index=True)

    return scoresdf


def main():
    options = tools.parse_options()
    start = time.time()
    if os.path.isfile(options.input):  # if results are already stored then use that as input
        scoresdf = pd.read_csv(options.input)
    else:  # in previous experiments, if results are not stored then create new dataframe to store the results
        scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier', 'Contrast_name', 'Balanced_accuracy'])

    mat_files = os.listdir(options.data)
    contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    n_back_list = list(filter(lambda x: 'nBack' in x and ('2' in x or '3' in x ), contrast_list))
    faces_list = list(filter(lambda x: 'Faces' in x and ('5' in x or '4' in x or '3' in x), contrast_list))
    relevant_contrast_list = n_back_list + faces_list  # extracted nBack 2,3 and Faces 3,4,5 contrasts

    # Age and gender information along with subject id is extracted
    file = open(options.additional_data + "/subject_name.txt", "r")
    ids = file.read().split()
    ids = [int(float(id)) for id in ids]
    edf = pd.read_csv(options.additional_data + '/n300.csv')
    edf['subject_cont'] = ids
    edf = edf[['KJØNN', 'subject_cont', 'ALDER']]
    edf = edf.rename(columns={'KJØNN': 'gender', 'ALDER': 'age'})

    for contrast in relevant_contrast_list:
        contrast_name = contrast.split(".")[0]
        if len(scoresdf[scoresdf["Contrast_name"] == contrast_name]):
            continue

        for nClass in range(2, 4, 1):
            #  Considering all classes: Bipolar, Schizo and Control
            if nClass == 3:
                df, contrast_name = tools.data_extraction(options.data, nClass, contrast, options.data_type)
                df = mlu.missing_values(df)
                df = pd.merge(df, edf, on=['subject_cont'], how='inner')
                scoresdf = run_no_gender_ml(df, options, 123, scoresdf, contrast_name)

            #  Considering combination of 2 classes: Bipolar-Schizo, Schizo-Control and Control-Bipolar
            elif nClass == 2:
                df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, contrast, options.data_type)

                # Combining two pairs off all combination
                df12 = df1.append(df2)
                df23 = df2.append(df3)
                df31 = df3.append(df1)

                # Handle missing values
                df12 = mlu.missing_values(df12)
                df23 = mlu.missing_values(df23)
                df31 = mlu.missing_values(df31)

                # Adding age and gender data for Standardization purpose. This additional data will be removed in
                # data preprocessing
                df12 = pd.merge(df12, edf, on=['subject_cont'], how='inner')
                df23 = pd.merge(df23, edf, on=['subject_cont'], how='inner')
                df31 = pd.merge(df31, edf, on=['subject_cont'], how='inner')


                scoresdf = run_no_gender_ml(df12, options, 12, scoresdf, contrast_name)
                scoresdf = run_no_gender_ml(df23, options, 23, scoresdf, contrast_name)
                scoresdf = run_no_gender_ml(df31, options, 31, scoresdf, contrast_name)


        scoresdf.to_csv(options.output + "individual.csv", index=False)

    print("It took %s seconds to run %s iterations for %s model after removing gender effect" % (time.time() - start, options.number_iterations,
                                                                    options.model))




if __name__ == '__main__':
    main()