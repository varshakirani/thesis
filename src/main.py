import sys
import time #to check total time took for running the script or function
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu

# Todo: change output variable to pandas dataFrame. Store as csv file and
# Todo: use the same for visualization.
# Todo: Remove old code and commit in the github

def run_basic_ml(df, number_iterations, model_option, kFold, n, scoresdf, contrast_name):

    models = ["svm_kernel", "naive_bayes", "decision_tree"]
    for i in range(number_iterations):
        train, test = mlu.train_test_split(df)
        X, y = mlu.get_features_labels(train)
        tX, ty = mlu.get_features_labels(test)

        if model_option == "all":
            for model_name in models:
                train_score, trained_model = mlu.model_fitting(model_name, X, y, kFold)
                test_score = trained_model.score(tX, ty)
                scoresdf = scoresdf.append(
                    {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name}, ignore_index=True)
                scoresdf = scoresdf.append(
                    {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name}, ignore_index=True)



        else:
            train_score, trained_model = mlu.model_fitting(model_name, X, y, kFold)
            test_score = trained_model.score(tX, ty)
            scoresdf = scoresdf.append(
                {'Score': train_score, 'Type': 'train', 'Model': 'svm_kernel', 'Classifier': n,
                 'Contrast_name':contrast_name}, ignore_index=True)
            scoresdf = scoresdf.append(
                {'Score': test_score, 'Type': 'test', 'Model': 'svm_kernel', 'Classifier': n,
                 'Contrast_name':contrast_name}, ignore_index=True)

    return scoresdf

def main():
    print("NI Thesis")
    options = tools.parse_options()
    start = time.time()
    scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier'])
    for nClass in range(2,4,1):

        if nClass == 3:
            # Read Data and put it into panda data frame. Initially considering only means
            df, contrast_name = tools.data_extraction(options.data, nClass)
            df = mlu.missing_values(df)
            print("ML on 123")
            scoresdf = run_basic_ml(df, options.number_iterations, options.model, options.kFold, 123, scoresdf,contrast_name)

        elif nClass == 2:
            df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass)
            # Combining two pairs off all combination
            df12 = df1.append(df2)
            df23 = df2.append(df3)
            df31 = df3.append(df1)

            # Handle missing values
            df12 = mlu.missing_values(df12)
            df23 = mlu.missing_values(df23)
            df31 = mlu.missing_values(df31)

            print("ML on 12")
            scoresdf = run_basic_ml(df12, options.number_iterations, options.model, options.kFold, 12, scoresdf , contrast_name)
            print("ML on 23")
            scoresdf = run_basic_ml(df23, options.number_iterations, options.model, options.kFold, 23, scoresdf ,contrast_name)
            print("ML on 31")
            scoresdf = run_basic_ml(df31, options.number_iterations, options.model, options.kFold, 31, scoresdf, contrast_name)

    print(scoresdf.shape)

    scoresdf.to_csv(options.output+"/without_kfold.csv", index=False)

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

