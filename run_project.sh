#!/usr/bin/env bash


#to run with missing values Mean Replacement
# --missing_data 1 -> mean replacement 2-> median replacement

#For without tuning
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/_without_tuning.csv" --missing_data=1

#For with tuning
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/_with_tuning.csv" --missing_data=1 -t

#For combined without tuning
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/combined_without_tuning.csv" --missing_data=1 -c

#For combined with tuning
#python src/main.py -d="../Data" -n=100 -o="out/output_scores_testing/" -i="out/output_scores_testing/combined_with_tuning.csv" --missing_data=1 -c -t


# Visualization

#python src/visualization.py -o="out/outputs_mv1/"
#python src/plotting.py -o="out/outputs_scores/" -i="out/outputs_scores/without_tuning.csv"

python src/plotting.py -o="out/output_scores_testing/plots/" -i="out/output_scores_testing/combined_with_tuning.csv"
