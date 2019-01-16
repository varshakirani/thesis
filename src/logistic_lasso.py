from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import l1_min_c
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
sys.path.insert(0, 'src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from time import time


def run_logistic_lasso(df, contrast_name, classifier_no, out):
    min_max_scaler = MinMaxScaler()

    clf_cv = linear_model.LogisticRegressionCV(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6),
                                               cv=10)
    coefs_ = []
    for i in range(100):
        X, y = mlu.get_features_labels(shuffle(df))
        X = min_max_scaler.fit_transform(X)
        clf_cv.fit(X, y)
        coefs_.append(clf_cv.coef_.ravel().copy())

    a = np.asarray(coefs_)
    np.savetxt(out + "%s_%s_logisticLasso.csv" % (contrast_name, classifier_no), a, delimiter=",")


if __name__ == '__main__':
    print("Logistic Regression with Lasso penalty")

    options = tools.parse_options()

    mat_files = os.listdir(options.data)
    contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    n_back_list = list(filter(lambda x: 'nBack' in x and ('2' in x or '3' in x), contrast_list))
    faces_list = list(filter(lambda x: 'Faces' in x and ('5' in x or '4' in x), contrast_list))
    relevant_mat_files = n_back_list + faces_list
    start = time()
    for mat_file in relevant_mat_files:
        print(mat_file)
        df1, df2, df3, contrast_name = tools.data_extraction(options.data, 2, mat_file, 'face_aal')
        df1 = shuffle(df1)
        df2 = shuffle(df2)
        df3 = shuffle(df3)

        # Combining two pairs off all combination
        df12 = df1.append(df2)
        df23 = df2.append(df3)
        df31 = df3.append(df1)

        # Handle missing values
        df12 = mlu.missing_values(df12)
        df23 = mlu.missing_values(df23)
        df31 = mlu.missing_values(df31)

        run_logistic_lasso(df12, contrast_name, 12, options.output)
        run_logistic_lasso(df23, contrast_name, 23, options.output)
        run_logistic_lasso(df31, contrast_name, 31, options.output)


    print("It took %0.3fs to do 100 times lasso on all contrasts" %(time()-start))


