import sys
import time #to check total time took for running the script or function
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os
import logging


# Create and configure the logger
# LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
# logging.basicConfig(filename='logs/running_ml.log', level=logging.DEBUG, format=LOG_FORMAT)
# logger = logging.getLogger()

# logger.info('Running all the ml models on all the contrast')
def run_basic_ml(df, options, n, scoresdf, contrast_name):
    print(contrast_name)
    #models = ["svm_kernel_default", "svm_kernel_tuned", "naive_bayes", "decision_tree", "rfc", 'logistic_regression']
    #models = ['svm_kernel_default',"svm_kernel_tuned"]
    models = ["svm_kernel_default","naive_bayes", "decision_tree"]#, "rfc", 'logistic_regression']
    for i in range(options.number_iterations):
        train, test = mlu.train_test_split(df)
        x_train, y_train = mlu.get_features_labels(train)
        x_test, y_test = mlu.get_features_labels(test)

        if options.model == "all":
            for model_name in models:
                #logger.debug("Running the %s model of the %s th iteration for %s contrast" %(model_name, i, contrast_name))
                train_score, train_balanced_score, trained_model= mlu.model_fitting(model_name, x_train, y_train, options.kFold)
                test_score = trained_model.score(x_test, y_test)
                test_balanced_score = mlu.balanced_accuracy(trained_model.predict(x_test),y_test)
                #print(model_name + " Train:"+ str(train_score) + "  Test:" +str(test_score) +" Contrast:" +contrast_name)
                scoresdf = scoresdf.append(
                    {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name, 'Balanced_accuracy':train_balanced_score}, ignore_index=True)
                scoresdf = scoresdf.append(
                    {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name,'Balanced_accuracy':test_balanced_score}, ignore_index=True)

        else:
            train_score, trained_model= mlu.model_fitting(options.model, x_train, y_train, options.kFold)
            test_score = trained_model.score(x_test, y_test)
            scoresdf = scoresdf.append(
                {'Score': train_score, 'Type': 'train', 'Model': options.model, 'Classifier': n,
                 'Contrast_name':contrast_name, 'Balanced_accuracy':train_balanced_score}, ignore_index=True)
            scoresdf = scoresdf.append(
                {'Score': test_score, 'Type': 'test', 'Model': options.model, 'Classifier': n,
                 'Contrast_name':contrast_name,'Balanced_accuracy':test_balanced_score}, ignore_index=True)

    return scoresdf


def main():
    print("NI Thesis")
    options = tools.parse_options()
    start = time.time()

    if options.combine:
        o_subtitle = 'combined'
    else:
        o_subtitle = 'individual'

    if os.path.isfile(options.input):
        scoresdf = pd.read_csv(options.input)
    else:
        scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier', 'Contrast_name', 'Balanced_accuracy'])

    contrast_list = ["Faces_con_0001.mat",'Faces_con_0001_389.mat','nBack_con_0001.mat','nBack_con_0001_407.mat' ]
    i = 0
    #for i in range(0, len(contrast_list),2): #TODO uncomment this for combined
    #for i in range(len(contrast_list)):
    for mat_file in os.listdir(options.data): #TODO uncomment this for individual

        #print(mat_file)
        #contrast_name = mat_file.split(".")[0]
        contrast_name = contrast_list[i].split(".")[0]
        # Checking if the training is already made for the particular contrast

        # TODO Uncomment this for checking if contrast is present in the file
        if len(scoresdf[scoresdf['Contrast_name'] == contrast_name]):
            continue

        for nClass in range(2,4,1):

            if nClass == 3:

                # Read Data and put it into panda data frame. Initially considering only means
                if options.combine:
                    #print(contrast_list[i], contrast_list[i+1])
                    df, contrast_name = tools.combine_contrast(options.data, nClass, contrast_list[i], contrast_list[i+1])
                    pass
                else:
                    df, contrast_name = tools.data_extraction(options.data, nClass, mat_file) #TODO uncomment this for individual
                    #df, contrast_name = tools.data_extraction(options.data, nClass, contrast_list[i]) #TODO uncomment this for combined
                df = mlu.missing_values(df)
                scoresdf = run_basic_ml(df, options, 123, scoresdf,contrast_name)


            elif nClass == 2:

                if options.combine:
                    #print(contrast_list[i], contrast_list[i + 1])
                    df1, df2, df3, contrast_name = tools.combine_contrast(options.data, nClass, contrast_list[i], contrast_list[i+1])
                    pass
                else:
                    df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, mat_file) #TODO uncomment this for individual
                    #df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, contrast_list[i]) #TODO uncomment this for combined
                # Combining two pairs off all combination
                df12 = df1.append(df2)
                df23 = df2.append(df3)
                df31 = df3.append(df1)

                # Handle missing values
                df12 = mlu.missing_values(df12)
                df23 = mlu.missing_values(df23)
                df31 = mlu.missing_values(df31)

                scoresdf = run_basic_ml(df12, options, 12, scoresdf ,contrast_name)
                scoresdf = run_basic_ml(df23, options, 23, scoresdf ,contrast_name)
                scoresdf = run_basic_ml(df31, options, 31, scoresdf, contrast_name)


        scoresdf.to_csv(options.output + "%s.csv" % (o_subtitle), index=False)


    #print(scoresdf.shape)

    #scoresdf.to_csv(options.output+"%s.csv"%(o_subtitle), index=False)

    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))


if __name__ == '__main__':
    main()

