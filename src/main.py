import sys
import time #to check total time took for running the script or function
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os

# Todo: change output variable to pandas dataFrame. Store as csv file and
# Todo: use the same for visualization.
# Todo: Remove old code and commit in the github


def run_basic_ml(df, options, n, scoresdf, contrast_name):

    models = ["svm_kernel", "naive_bayes", "decision_tree"]

    for i in range(options.number_iterations):
        train, test = mlu.train_test_split(df)
        x_train, y_train = mlu.get_features_labels(train)
        x_test, y_test = mlu.get_features_labels(test)

        if options.model == "all":
            for model_name in models:
                train_score, trained_model = mlu.model_fitting(model_name, x_train, y_train, options.kFold,  options.tuning,)
                test_score = trained_model.score(x_test, y_test)
                scoresdf = scoresdf.append(
                    {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name}, ignore_index=True)
                scoresdf = scoresdf.append(
                    {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name}, ignore_index=True)

        else:
            if len(scoresdf[(scoresdf['Contrast_name'] == contrast_name) & (scoresdf['Model'] == model_name)
                    & (scoresdf['Tuning'] == options.tuning)]):
                continue
            train_score, trained_model = mlu.model_fitting(model_name, x_train, y_train, options.kFold,  options.tuning,)
            test_score = trained_model.score(x_test, y_test)
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

    o_subtitle = 'without_tuning'
    if options.tuning:
        o_subtitle = 'with_tuning'


    if os.path.isfile(options.output + '%s.csv'%(o_subtitle)):
        scoresdf = pd.read_csv(options.output + '%s.csv' % (o_subtitle))
    else:
        scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier', 'Contrast_name'])

    for mat_file in os.listdir(options.data):
        print(mat_file)
        contrast_name = mat_file.split(".")[0]
        # Checking if the training is already made for the particular contrast

        if len(scoresdf[scoresdf['Contrast_name'] == contrast_name]):
            continue

        for nClass in range(2,4,1):

            if nClass == 3:
                # Read Data and put it into panda data frame. Initially considering only means
                df, contrast_name = tools.data_extraction(options.data, nClass, mat_file)
                df = mlu.missing_values(df)
                #print("ML on 123")
                scoresdf = run_basic_ml(df, options, 123, scoresdf,contrast_name)


            elif nClass == 2:
                df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, mat_file)
                # Combining two pairs off all combination
                df12 = df1.append(df2)
                df23 = df2.append(df3)
                df31 = df3.append(df1)

                # Handle missing values
                df12 = mlu.missing_values(df12)
                df23 = mlu.missing_values(df23)
                df31 = mlu.missing_values(df31)

                #print("ML on 12")
                scoresdf = run_basic_ml(df12, options, 12, scoresdf ,contrast_name)
                #print("ML on 23")
                scoresdf = run_basic_ml(df23, options, 23, scoresdf ,contrast_name)
                #print("ML on 31")
                scoresdf = run_basic_ml(df31, options, 31, scoresdf, contrast_name)

        scoresdf.to_csv(options.output + "%s.csv" % (o_subtitle), index=False)

    print(scoresdf.shape)

    #scoresdf.to_csv(options.output+"%s.csv"%(o_subtitle), index=False)

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

