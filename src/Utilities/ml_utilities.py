import math
import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np


def train_test_split(df):
    """
    :param df: panda dataFrame which has means of ROIs
    :return: Training data set and testing dataset
    """

    df = shuffle(df)

    split_index = math.floor(0.8 * len(df))
    train = df[:split_index]
    test = df[split_index:]

    return train, test


def balanced_accuracy(predictions, true_values):
    label1 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 1, predictions, true_values))
    label2 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 2, predictions, true_values))
    label3 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 3, predictions, true_values))
    true_label1 = float(sum(true_values == 1))
    true_label2 = float(sum(true_values == 2))
    true_label3 = float(sum(true_values == 3))
    if not true_label1 :
        true_label1 = 1.0
    if not true_label2:
        true_label2 = 1.0
    if not true_label3:
        true_label3 = 1.0


    baccuracy = ((sum(label1) / true_label1) +
                 (sum(label2) / true_label2) +
                 (sum(label3) / true_label3)) / 3

    return baccuracy


def model_fitting(model_name, X, y, kFold=10):
    if model_name == "svm_kernel_default":
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
    elif model_name == "svm_kernel_tuned":
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 2 ** -5, 2 ** -10, 2 ** 5], 'kernel': ['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, cv=kFold)
        grid.fit(X, y)
        best_param = grid.best_params_
        model = svm.SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
    elif model_name == "naive_bayes":
        model = GaussianNB()
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name =="svm_linear":
        model = svm.SVC(kernel="linear")
    elif model_name == "rfc":
        model = RandomForestClassifier(n_estimators=200)
    elif model_name == "logistic_regression":
        model = LogisticRegression()
    else:
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)

    model.fit(X, y)
    scores = model.score(X, y)
    pred = model.predict(X)
    balanced_accuracy_score = balanced_accuracy(pred,y)

    return scores, balanced_accuracy_score ,model


def model_test(test, model):
    # Testing
    test_data = test.loc[:, test.columns != "label"].values
    test_actual_results = np.asarray(test.label).astype(float)
    test_prediction = model.predict(test_data)
    total_test_samples = test_data.shape[0]
    total_correct_predictions = np.count_nonzero(test_actual_results == test_prediction)
    test_accuracy = np.asarray(total_correct_predictions / total_test_samples)
    #print("Test Accuracy is {}.".format(test_accuracy))
    #print(test_accuracy)
    return test_accuracy


def get_features_labels(data):
    """
    :param data: pandas dataframe which has label, subject_cont and ROI mean values
    :return: X: Features and y: corresponding labels

    """
    p = data.drop('subject_cont', axis = 1)
    X = p.loc[:, p.columns != "label"].values
    y = np.asarray(data.label).astype(int)

    return X,y


def missing_values(df, method=1):
    """
    This function replaces NaN in the df with 0. This is temporary fix.
    @TODO: Should do research for better method to handle missing solutions
    :param df: panda dataFrame which has means of ROIs
    :param method: 0 for filling it with 0s, 1 for mean replacement
    :return: panda dataFrame which has means of ROIs with no missing values
    """
    # TODO: Should do research for better method to handle missing solutions

    # np.all(np.isfinite(df))  # to check if there is any infinite number
    # np.any(np.isnan(df))  # to check if there is any nan
    # np.where(np.asanyarray(np.isnan(df)))  # to find the index of nan

    if method == 1:
        # Mean replacement for nan numbers in particular label with the mean of non missing values in the same label.
        df.fillna(df.mean(), inplace=True)

    elif method == 2:
        # Median replacement for nan numbers in particular label with the Median of non missing values in the same label
        df.fillna(df.median(), inplace=True)
    else:

        df.fillna(0, inplace=True)
    return df
