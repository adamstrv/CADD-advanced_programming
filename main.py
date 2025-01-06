import torch
import csv
import pandas as pd
import torch

from train_network import scaler
from train_network import pca
from train_network import trained_model

"""
This script uses the trained model to predict the classes (0 = inactive or 1 = ative) of the test molecules
and writes it to a csv. file that we can submit directly.
"""


test_frame = pd.read_csv('testing_dataframe.csv')                                        #  Define the test frame with info (predictors) of the smiles that we already extracted from the ogirinal file using 'data_extraction.py'

ID = test_frame['ID'].tolist()                                                           #  Define a list of the ID's (so we can couple them again later)
X = test_frame.drop('ID', axis = 'columns').drop('Unnamed: 0', axis='columns')           #  Define X as the predictors

X = scaler.transform(X)                                                                  #  Scale the data the same way we scaled the training data using the scaler we defined in 'train_network.py'
reduced_X_frame = pca.transform_frame(X)                                                 #  Use the same dimensionality reduction as on the training data (95% of the variance) using the pca we defined in 'train_network.py'
X_test = torch.tensor(reduced_X_frame.values, dtype=torch.float32)                       #  Convert to a tensor so it can be interpreted by the model

Y_pred = trained_model.predict(X_test)                                                   #  Use the model to predict 0/1 for

with open("Submission_file_v4_group1.csv", mode='w', newline="") as csvfile:             #  Write the results to a csv. file
    fieldnames = ['Unique_ID', 'target_feature']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(len(ID)):
        writer.writerow({'Unique_ID':int(ID[i]), 'target_feature': str(Y_pred[i])})