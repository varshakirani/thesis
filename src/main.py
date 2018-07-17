import sys
from sklearn.utils import shuffle
import time #to check total time took for running the script or function

sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')

import tools
import ml_utilities as mlu


class Result:
    """
    Result contains the cross validation scores of the respective machine learning algorithms which are run through
    number of iterations.
    """

    out = {}

    def __init__(self, svm, naive_bayes, decision_tree):

        self.out["svm"] = svm
        self.out["naive_bayes"] = naive_bayes
        self.out["decision_tree"] = decision_tree

        #def append_svm_scores(self, svm = []):
        #self.svm.append(svm)


def run_basic_ml(df, number_iterations, model_option, kFold,n, dump=False, output="."):
    train, test = mlu.train_test_split(df)
    models = ["svm", "naive_bayes", "decision_tree"]

    result_scores = Result([], [], [])
    for i in range(number_iterations):
        # print("Running iteration :%s  " % (i + 1))
        X, y = mlu.get_features_labels(shuffle(train))
        if model_option == "all":
            for model in models:
                scores = mlu.model_fitting(model, X, y, kFold)
                result_scores.out[model].append(scores)
        else:
            scores = mlu.model_fitting(model_option, X, y, kFold)
            result_scores.out["svm"].append(scores)
    if dump:
        for key, value in result_scores.out.items():
            tools.dump_results_to_json(key, value, output, n)


def main():
    print("NI Thesis")
    options = tools.parse_options()
    start = time.time()

    if options.nClass == 3:
        # Read Data and put it into panda data frame. Initially considering only means
        df = tools.data_extraction(options.data, options.nClass)
        df = mlu.missing_values(df)
        run_basic_ml(df, options.number_iterations, options.model, options.kFold, "123", True, options.output)

    elif options.nClass == 2:
        # Done: Read the data and separate into 3 different dataframes.
        # Done: For each pair combination combine it, shuffle it
        # Done: Randomnly split it into training and testing set
        # Done: Fit the models and write the scores into JSON files
        # Done: Visualization should be fixed for that new JSON files

        print("biClass Classification for every pair among Bipolar(1), Schizophrenia(2) and Control(3)")

        df1, df2, df3 = tools.data_extraction(options.data, options.nClass)
        # Combining two pairs off all combination
        df12 = df1.append(df2)
        df23 = df2.append(df3)
        df31 = df3.append(df1)
        # Handle missing values
        df12 = mlu.missing_values(df12)
        df23 = mlu.missing_values(df23)
        df31 = mlu.missing_values(df31)
        run_basic_ml(df12, options.number_iterations, options.model, options.kFold, "12", True, options.output)
        run_basic_ml(df23, options.number_iterations, options.model, options.kFold, "23", True, options.output)
        run_basic_ml(df31, options.number_iterations, options.model, options.kFold, "31", True, options.output)

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

