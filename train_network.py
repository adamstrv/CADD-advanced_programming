from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import torch
from PCA import custom_PCA
from neural_network_V2 import NN_BinClass

import pandas as pd

train_frame = pd.read_csv('training_dataframe.csv')
Y = train_frame['classification']                                                                        #   Define Y with the classification
X = train_frame.drop('classification', axis = 'columns').drop('Unnamed: 0', axis='columns')              #   Define X with all the features

scaler = StandardScaler()
scaler.fit(X)                                                                     #   Scale the data
X = scaler.transform(X)

pca = custom_PCA(X)                                                               #   Define a 'dimensionality-reduced' X and save the pca for new variables
reduced_X_frame = pca.transform_frame(X)

X_tensor = torch.tensor(reduced_X_frame.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32)


if __name__ == "__main__":

    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2)          # Define test and training data

    train_dataset = TensorDataset(X_train,Y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TensorDataset(X_val,Y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    num_features = X_train.shape[1]

    model = NN_BinClass(input_size=num_features)

    model.training_loop(train_loader, val_loader, num_epochs=500, filename= "saved_model_v8.pth")
    best_loss = model.best_val_loss
    model.show_loss_curves()

trained_model = NN_BinClass()
trained_model.load_state_dict(torch.load("saved_model_v6.pth", weights_only=True))