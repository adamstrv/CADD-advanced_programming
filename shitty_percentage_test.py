import torch
import pandas as pd

from train_network import scaler
from train_network import pca
from train_network import trained_model


train_frame = pd.read_csv("datasets/training_dataframe.csv")

Y = train_frame['classification'].tolist()                                                           #   Define Y with the classification
X = train_frame.drop('classification', axis = 'columns').drop('Unnamed: 0', axis='columns')          #   Define X with all the features

X = scaler.transform(X)
reduced_X_frame = pca.transform_frame(X)
X_test = torch.tensor(reduced_X_frame.values, dtype=torch.float32)
Y_pred = trained_model.predict(X_test)

count = 0
for i in range(len(Y)):
    if Y[i] == Y_pred[i]:
        count += 1

percentage = count/len(Y)*100

print(percentage, '%')