#!/usr/bin/env bash


#TODO to run with missing values Mean Replacement
# --missing_data 1 -> mean replacement 2-> median replacement

#TODO For without tuning
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/_without_tuning.csv" --missing_data=1

#TODO For combined
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/combined_without_tuning.csv" --missing_data=1 -c

#TODO Combined Contrasts.For single output file. Here we have removed tuning parameter. So the data is stored in single file
#python src/main.py -d="../Data" -n=1 -o="out/svm_check/" -i="out/svm_check/combined.csv" --missing_data=1 -c

#TODO Individual Contrasts.For single output file. Here we have removed tuning parameter. So the data is stored in single file
python src/main.py -d="../Data" -n=100 -o="out/normalized/" -i="out/normalized/individual.csv" --missing_data=1 --normalize

#TODO Visualization

#python src/visualization.py -o="out/outputs_mv1/"
#python src/plotting.py -o="out/outputs_scores/" -i="out/outputs_scores/without_tuning.csv"

#python src/plotting.py -o="out/output_scores_testing/plots/" -i="out/output_scores_testing/combined_with_tuning.csv" -b
#python src/plotting.py -o="out/single_output_file/plots/" -i="out/single_output_file/combined.csv" --type=1 -b
#python src/plotting.py -o="out/single_output_file/plots/" -i="out/single_output_file/individual.csv" --type=1 -b
#python src/plotting.py -o="out/without_tuning/plots/" -i="out/without_tuning/individual.csv" --type=1 -b
#python src/plotting.py -o="out/svm_check/plots/" -i="out/svm_check/individual.csv" --type=1 -b
#python src/plotting.py -o="out/single_output_file/plots_new/" -i="out/single_output_file/individual.csv" --type=1 -p

#TODO exploration of the data

#python src/data_exploration.py -m