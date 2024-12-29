import torch
import csv
import pandas as pd
import torch

from train_network import scaler
from train_network import pca
from train_network import trained_model

"""
this script uses the trained model to predict the classes of the test smiles
and writes it to a csv. file that we can submit directly.
"""


test_frame = pd.read_csv('testing_dataframe.csv')                                        #  Define test frame

ID = test_frame['ID'].tolist()                                                           #  Define a list of the ID's
X = test_frame.drop('ID', axis = 'columns').drop('Unnamed: 0', axis='columns')           #  Define X as the predictors

X = scaler.transform(X)                                                                  #  Scale the data the same way we scaled the training data
reduced_X_frame = pca.transform_frame(X)                                                 #  Use the same dimensionality reduction as on the training data
X_test = torch.tensor(reduced_X_frame.values, dtype=torch.float32)

Y_pred = trained_model.predict(X_test)                                                   #  Use the model to predict an output

with open("submission_files\Submission_file_v4_group1.csv", mode='w', newline="") as csvfile:             #  Write the results to a csv. file
    fieldnames = ['Unique_ID', 'target_feature']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(len(ID)):
        writer.writerow({'Unique_ID':int(ID[i]), 'target_feature': str(Y_pred[i])})