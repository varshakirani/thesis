import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_avg_original_accuracy(raw_folder):
    original_files = list( filter(lambda x: x.startswith('permutation_result_'), os.listdir(raw_folder)))
    result = pd.DataFrame(columns=['contrast', 'class', 'Model', 'Avg original accuracy'])
    for file in original_files:
        df = pd.read_csv(raw_folder+file)
        res = pd.DataFrame({"Avg original accuracy":
                        df.groupby(['contrast','class','Model'])['original_accuracy'].mean()}).reset_index()
        result = pd.concat([res,result], ignore_index=True)
    return result


def calculate_pvalue(raw_folder, df_orig):
    per_files = list(filter(lambda x: x.startswith('n') or x.startswith('F'), os.listdir(raw_folder)))
    contrast_names = {
        'n2' : 'nBack_con_0002',
        'n3' : 'nBack_con_0003',
        'F3' : 'Faces_con_0003',
        'F4' : 'Faces_con_0004',
        'F5' : 'Faces_con_0005'
    }
    df_pvalues = pd.DataFrame(columns=['contrast', 'Model', 'class', "Avg original accuracy", "p_value"])
    for file in per_files:
        contrast = contrast_names[file[0:2]]
        classifier = file[3:5]
        model = file[6:].split(".")[0]

        original_accuracy = df_orig[(df_orig['contrast'] == contrast) & (df_orig['Model'] == model)
                                        & (df_orig['class'] == int(classifier) )]["Avg original accuracy"].values[0]

        df_perm = pd.read_csv(raw_folder+file, header=None, names=['performance_scores'])

        #performance_scores = np.asarray(df_perm['performance_scores'])
        #p_value = sum(performance_scores >= original_accuracy)/10000
        p_value = df_perm[df_perm['performance_scores'] >= original_accuracy]['performance_scores'].count()/10000
        df_pvalues = df_pvalues.append({'contrast': contrast,
                                        "Model":model,
                                        "class":int(classifier),
                                        "Avg original accuracy":original_accuracy,
                                       "p_value":p_value},
                                       ignore_index=True)
    return df_pvalues




def plot_permutation(df, output):
    df = df.sort_values("Model")

    all_c = df['contrast'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)

    print(face_c)
    print(nback_c)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))

    axs = axes.ravel()

    y_val = 'p_value'
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 12) &
                          (df['contrast'].isin(face_c))], ax=axs[0]).set_title('Faces Bipolar and Schizo')

    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 23) &
                          (df['contrast'].isin(face_c))], ax=axs[1]).set_title('Faces Schizo and Control')
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 31) &
                          (df['contrast'].isin(face_c))], ax=axs[2]).set_title('Faces Control and Bipolar')
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 12) &
                          (df['contrast'].isin(nback_c))], ax=axs[3]).set_title('nBack Bipolar and Schizo')
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 23) &
                          (df['contrast'].isin(nback_c))], ax=axs[4]).set_title('nBack Schizo and Control')
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 31) &
                          (df['contrast'].isin(nback_c))], ax=axs[5]).set_title('nBack Control and Bipolar')

    for i in range(6):
        axs[i].set_xticklabels(['lr', 'rfc', 'rbf_b', 'rbf'])
        axs[i].set_xlabel('')
        axs[i].set_ylim(0.0001,1)
        axs[i].set(yscale="log")
        if i == 2 or i == 5:
            continue
        axs[i].legend_.remove()
    for i in range(3, 6, 1):
        axs[i].set_xlabel('Models')
    fig.suptitle('Significance of model classification accuracy using Permutation test after removal of gender information', fontsize=22)
    plt.savefig("%sOverallPerformance.png" % (output))
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == "__main__":

    raw_folder = "Ultimate_final_output/no_gender/permutation/"

    df = get_avg_original_accuracy(raw_folder)
    df_pvalues = calculate_pvalue(raw_folder, df)
    plot_permutation(df_pvalues, raw_folder)
