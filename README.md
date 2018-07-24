# Project Description
This project analyses fMRI mean values of face contrasts of Bipolar 
disorder, Schizophrenia and Control subjects inorder to find the 
vital ROI responsible for respective subjects.

# Install Packages
#### Packages required for this repo to work 
* scipy
* pandas
* scikit-learn
* matplotlib
* seaborn

These can be installed by running the following command

```console
$ bash install_packages.sh
```
# Running Scripts

Run the following commands inside src folder
#### Inorder to obtain the cross validation scores:  


```console
$ python src/main.py -m=MODEL_NAME -n=NUMBER_OF_ITERATIONS -d=INPUT_FOLDER -k=kFOLD_NUMBER
```

#### For plotting the histograms
```console
$ python src/visualization.py -o=OUTPUT_FOLDER
```
###### To run with default flags
```console
$ python src/main.py
$ python src/visualization.py
``` 
