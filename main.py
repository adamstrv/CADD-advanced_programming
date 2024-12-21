from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

from data_extraction import extract_train_smiles
from extract_descriptor_types import extract_descriptor_types
from neural_network import NN_BinClass

from train_network import scaler
from train_network import pca


smiles = extract_train_smiles('shortertrain.csv')                                 #   Extract a library with data from the a csv file
train_descriptor_frame = extract_descriptor_types(smiles)                         #   Extract a dataframe with features of each smile via RDkit
Y = train_descriptor_frame['classification']                                      #   Define Y with the classification
X = train_descriptor_frame.drop('classification', axis = 'columns')               #   Define X with all the features
X = scaler.transform(X)                                                           #   Scale the data
reduced_X_frame = pca.transform_frame(X)                                          #   Define a 'dimensionality-reduced' X and save the pca for new variables


X_tensor = torch.tensor(reduced_X_frame.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2)          # Define test and training data

train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_val,Y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

num_features = X_train.shape[1]

# evaluate the best model on the test dataset
#trained_model = NN_BinClass(num_features)
#trained_model.load_state_dict(torch.load('best_model.pth',weights_only=True))

#test_loss, test_accuracy = trained_model.evaluate(val_loader)
#print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')