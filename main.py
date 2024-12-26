import torch
import csv
import pandas as pd
import torch

from train_network import scaler
from train_network import pca
from train_network import trained_model


test_frame = pd.read_csv('testing_dataframe.csv')

ID = test_frame['ID'].tolist()
X = test_frame.drop('ID', axis = 'columns') 

X = scaler.transform(X)
reduced_X_frame = pca.transform_frame(X)
X_test = torch.tensor(reduced_X_frame.values, dtype=torch.float32)

Y_pred = trained_model.predict(X_test)

with open("Submission_file_v2_group1.csv", mode='w', newline="") as csvfile:
    fieldnames = ['Unique_ID', 'target_feature']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(len(ID)):
        writer.writerow({'Unique_ID':int(ID[i]), 'target_feature': str(Y_pred[i])})