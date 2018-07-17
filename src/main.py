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


def main():
    print("NI Thesis")
    options = tools.parse_options()
    start = time.time()

    if options.nClass == 3:
        # Read Data and put it into panda data frame. Initially considering only means
        df = tools.data_extraction(options.data, options.nClass)
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
            tools.dump_results_to_json(key, value, options.output)

    elif options.nClass == 2:
        # Done: Read the data and separate into 3 different dataframes.
        # TODO: For each pair combination combine it, shuffle it
        # TODO: Randomnly split it into training and testing set
        # TODO: Fit the models and write the scores into JSON files
        # TODO: Visualization should be fixed for that new JSON files

        print("biClass Classification for every pair among Bipolar(1), Schizophrenia(2) and Control(3)")

        df1, df2, df3 = tools.data_extraction(options.data, options.nClass)
        # Combining two pairs off all combination
        df12 = shuffle(df1.append(df2))
        df23 = shuffle(df2.append(df3))
        df31 = shuffle(df3.append(df1))

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

