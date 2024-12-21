from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_extraction import extract_train_smiles
from torch.utils.data import DataLoader, TensorDataset
import torch
from extract_descriptor_types import extract_descriptor_types
from PCA import custom_PCA
from neural_network import NN_BinClass


scaler = StandardScaler()

train_smiles = extract_train_smiles('train.csv')                           #   Extract a library with data from the a csv file
train_descriptor_frame = extract_descriptor_types(train_smiles)                   #   Extract a dataframe with features of each smile via RDkit
Y = train_descriptor_frame['classification']                                      #   Define Y with the classification
X = train_descriptor_frame.drop('classification', axis = 'columns')               #   Define X with all the features

scaler.fit(X)                                                                     #   Scale the data
X = scaler.transform(X)
pca = custom_PCA(X)                                                               #   Define a 'dimensionality-reduced' X and save the pca for new variables
reduced_X_frame = pca.transform_frame(X)

X_tensor = torch.tensor(reduced_X_frame.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2)          # Define test and training data

train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_test,Y_test)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

num_features = X_train.shape[1]


if __name__ == "__main__":

    model = NN_BinClass(num_features)
    model.training_loop(train_loader, val_loader, num_epochs=150)
    model.show_loss_curves()


# evaluate the best model on the test dataset
trained_model = NN_BinClass(num_features)
trained_model.load_state_dict(torch.load('best_model.pth',weights_only=True))


#test_loss, test_accuracy = trained_model.evaluate(val_loader)
#print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')