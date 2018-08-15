import sys
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')
import time #to check total time took for running the script or function
import tools
import ml_utilities as mlu
from sklearn import svm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def main():
    input = "../Data"
    df,contrast_name = tools.data_extraction(input, 3, "Faces_con_0001.mat")
    df.fillna(df.mean(), inplace=True)

    scoresdf = pd.DataFrame(columns=['Score','Type','Model', 'Classifier'])


    # Model : model name

    for i in range(1):
        train, test = mlu.train_test_split(df)
        X, y = mlu.get_features_labels(train)
        tX, ty = mlu.get_features_labels(test)
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
        model.fit(X, y)
        train_score = model.score(X,y)
        test_score = model.score(tX, ty)
        predictions = model.predict(tX)
        print(len(ty))
        print(confusion_matrix(ty,predictions))
        print(classification_report(ty,predictions))
        param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001,0.00001, 2 ** -5, 2 ** -10, 2 ** 5 ],'kernel':['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, cv=10)
        grid.fit(X,y)
        best_param = grid.best_params_
        print((best_param))
        grid_predictions = grid.predict(tX)
        print(confusion_matrix(ty,grid_predictions))
        print(classification_report(ty,grid_predictions))

        ### finding scores after hyperparamter tuning
        model = svm.SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
        model.fit(X, y)
        train_score = model.score(X, y)
        test_score = model.score(tX, ty)
        scoresdf = scoresdf.append({'Score': train_score, 'Type': 'train', 'Model': 'svm_kernel','Classifier':123,'Contrast_name':contrast_name}, ignore_index=True)
        scoresdf = scoresdf.append({'Score': test_score, 'Type': 'test', 'Model': 'svm_kernel','Classifier':123 ,'Contrast_name':contrast_name}, ignore_index=True)


    fig, axes = plt.subplots(nrows=2, ncols=2)
    axs = axes.ravel()
    for j in range(4):

        models = scoresdf['Model'].unique()
        sns.boxplot( x='Model', y = 'Score', data=scoresdf[(scoresdf['Type'] == 'test') & (scoresdf['Model'] == 'svm_kernel')],ax=axs[j])
        #ax = sns.boxplot(x='Model',y='Score', hue='Type',data=scoresdf)






if __name__ == "__main__":
    main()