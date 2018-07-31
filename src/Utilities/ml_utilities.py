import math
import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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

def model_fitting(model_name, X, y, kFold = 10):
    if model_name == "svm":
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
    elif model_name == "naive_bayes":
        model = GaussianNB()
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name =="svm_linear":
        model = svm.SVC(kernel="linear")
    else:
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)

    cv = StratifiedKFold(kFold)
    model.fit(X, y)
    #scores = cross_val_score(model, X, y, cv=cv)
    scores = model.score(X, y)
    #print("Model: %s, Accuracy: %0.2f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))

    return scores, model


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

def get_features_labels(train):
    """
    :param train: Training dataset
    :return: X: Features and y: corresponding labels

    """
    X = train.loc[:, train.columns != "label"].values
    y = np.asarray(train.label)

    return X,y


def missing_values(df, method=0):
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
        tdf = pd.DataFrame(columns=df.columns)
        for l in df.label.unique():
            tdf = tdf.append(df[df.label == l].fillna(df[df.label == l].mean()))
        df = tdf
    elif method == 2:
        # Mean replacement for nan numbers in particular label with the mean of non missing values in the same label.
        tdf = pd.DataFrame(columns=df.columns)
        for l in df.label.unique():
            tdf = tdf.append(df[df.label == l].fillna(df[df.label == l].median()))
        df = tdf
    else:
        df = df.fillna(0) # replace nan with 0

    return df
