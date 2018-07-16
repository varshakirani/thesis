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

    svm = []
    naive_bayes = []
    decision_tree = []

    out = {}
    def __init__(self, svm, naive_bayes, decision_tree):
        #self.svm = svm
        #self.naive_bayes = naive_bayes
        #self.decision_tree = decision_tree

        self.out["svm"] = svm
        self.out["naive_bayes"] = naive_bayes
        self.out["decision_tree"] = decision_tree

    def append_svm_scores(self, svm = []):
        self.svm.append(svm)

    def append_naive_bayes_scores(self, naive_bayes = []):

        self.naive_bayes.append(naive_bayes)

    def append_decision_tree_scores(self, decision_tree = []):

        self.decision_tree.append(decision_tree)


def main():
    print("NI Thesis")
    options = tools.parse_options()

    if options.all:
        print("Running all the algorithms")

    if options.svm:
        print("Running SVM algorithm")

    if options.naive_bayes:
        print("Running naive bayes algorithm")
    start = time.time()
    # Read Data and put it into panda data frame. Initially considering only means
    df = tools.data_extraction(options.data)
    df = mlu.missing_values(df)
    train, test = mlu.train_test_split(df)
    models = ["svm", "naive_bayes", "decision_tree"]

    result_scores = Result([],[],[])
    for i in range(options.number_iterations):
        print("Running iteration :%s  " %(i+1))
        X, y = mlu.get_features_labels(shuffle(train))
        if options.model == "all":
            for model in models:
                scores = mlu.model_fitting(model, X, y, options.kFold)
                result_scores.out[model].append(scores)
        else:
            scores = mlu.model_fitting(options.model, X, y, options.kFold)
            result_scores.out["svm"].append(scores)

    for key, value in result_scores.out.items():
        tools.dump_results_to_json(key, value)

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

