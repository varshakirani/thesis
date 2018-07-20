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
    else:
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)

    cv = StratifiedKFold(kFold)
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=cv)
    #print("Model: %s, Accuracy: %0.2f (+/- %0.2f)" % (model_name, scores.mean(), scores.std() * 2))

    return scores


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
