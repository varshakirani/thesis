#!/usr/bin/env bash


# To run with missing values temp fix
# python src/main.py -d="../Data" -n=100 -o="src/outputs/" --nClass=3

#to run with missing values Mean Replacement
# --missing_data 1 -> mean replacement 2-> median replacement
 python src/main.py -d="../Data" -n=100 -o="src/outputs_mv2/" --nClass=2 --missing_data=1

# Visualization

#python src/visualization.py -o="src/outputs_mv2/"