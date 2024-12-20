import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_extraction import extract_train_smiles
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch import nn

from extract_descriptor_types import extract_descriptor_types
from PCA import custom_pca_fit


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_size, round(input_size/2))
        self.sigmoid1 = nn.Sigmoid() 
        self.linear2 = nn.Linear(round(input_size/2), round(input_size/4))        
        self.sigmoid2 = nn.Sigmoid()       
        self.output = nn.Linear(round(input_size/4), output_size)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):   
        x = self.linear1(x)
        x = self.sigmoid1(x)
    #    x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
    #    x = self.dropout(x)
        x = self.output(x)
        out = self.sigmoid(x)
        
        return out


train_smiles = extract_train_smiles('train.csv')                           # Extract a library with data from the a csv file
train_descriptor_frame = extract_descriptor_types(train_smiles)                   # Extract a dataframe with features of each smile via RDkit

Y = train_descriptor_frame['classification']                                      # Define Y with the classification
X = train_descriptor_frame.drop('classification', axis = 'columns')               # Define X with all the features

X = StandardScaler().fit_transform(X)                                             # Scale the data
reduced_X_frame, pca = custom_pca_fit(X)                                          # Define a 'dimensionality-reduced' X and save the pca

X_tensor = torch.tensor(reduced_X_frame.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2)          # Define test and training data
print(reduced_X_frame)

train_dataset = TensorDataset(X_train,Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_test,Y_test)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


num_features = X_tensor.shape[1]

print(num_features)

model = NeuralNetwork(num_features, 1)


criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.95)


# training function
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch in loader:
        X, Y = batch
        # convert labels to float for BCELoss            
        Y = Y.float()
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)        
        loss.backward()        
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

# evaluation function
def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            X, Y = batch
            # convert labels to float for BCELoss            
            Y = Y.float()        
        
            outputs = model(X)
            loss = criterion(outputs, X)
            
            running_loss += loss.item()            
            # apply threshold to get binary predictions
            predicted = (outputs > 0.5).float()  
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

# training loop
num_epochs = 50
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = evaluate(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%')
    
    # Check for the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')

