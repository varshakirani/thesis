import sys
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')

import tools
import ml_utilities as mlu






def main():
    print("NI Thesis")
    options = tools.parse_options()

    if options.all:
        print("Running all the algorithms")

    if options.svm:
        print("Running SVM algorithm")

    if options.naive_bayes:
        print("Running naive bayes algorithm")


    #Read Data and put it into panda datframe. Initially considering only means

    df = tools.data_extraction(options.data)
    df = mlu.missing_values(df)
    train, test = mlu.train_test_split(df)
    X,y = mlu.get_features_labels(train)

    models = ["svm","naive_bayes","decision_tree"]
    if(options.model == "all"):
        for model in models:
            mlu.model_fitting(model, X, y , options.kFold)
    else:
        mlu.model_fitting(options.model, X, y, options.kFold)





if __name__ == '__main__':
    main()

