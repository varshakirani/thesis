#Project Description
This project analyses fMRI mean values of face contrasts of Bipolar 
disorder, Schizophrenia and Control subjects inorder to find the 
vital ROI responsible for respective subjects.

#Running Scripts

Run the following command inside src folder
####Inorder to obtain the cross validation scores:  


```console
$ python main.py -m=MODEL_NAME -n=NUMBER_OF_ITERATIONS -d=INPUT_FOLDER -k=kFOLD_NUMBER
```

####For plotting the histograms
```console
$ python visualization.py -o=OUTPUT_FOLDER
```
###### To run with default flags
```console
$python main.py
$python visualization.py
``` 
