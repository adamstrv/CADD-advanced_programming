import torch
import torch.nn as nn
"""
Defines the class NN_BinClass, which is a neural network for binary classification.
"""

class NN_BinClass(nn.Module):
    """
    Defines the class NN_BinClass, which is a neural network for binary classification, with complete freedom to shape the model when initializing a model.
    When initializing the model, you can define the activation function, input size, the numer of layers, the number of neurons in the first layer, the reduction of
    neurons through the model (example: n_neurons = 20 and neuron reduction = 0.5, the first layer has 20 neurons, the second = 10, the third = 5, etc.), the dropout rate,
    the learning rate and the momentum of the gradient descent.

    With the training loop, you train the model using the training data, and save the best model then evaluating on the evaluation data, which it split beforehand.

    With the predict function, you predict the class of a new set of molecules
    """
    def __init__(self, activation = nn.Softsign, input_size = 80 , n_layers=2, n_neurons = 10, neuron_reduction = 1.0, dropout_rate = 0.7, learning_rate = 0.01, momentum = 0.6):
        super().__init__()
        self.layers = []
        self.acts = []
        self.dropout = []

        for i in range(n_layers):                                                       #   add a layer of the amount of layers
            self.layers.append(nn.Linear(input_size, n_neurons))
            self.acts.append(activation())
            self.dropout.append(nn.Dropout(dropout_rate))
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
            input_size = n_neurons
            n_neurons = round(neuron_reduction*n_neurons)
        self.linear_out = nn.Linear(n_neurons, 1)
        self.sigmoid = nn.Sigmoid()
    

        self.lr = learning_rate
        self.momentum = momentum

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), self.lr , self.momentum)

    def forward(self, x):
        """
        Just run the model forward, use the predictors of X to calculate an output.
        """
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.linear_out(x)
        out = self.sigmoid(x)
        return out


    def train_one_opoch(self, loader):
        """
        Train the model for one epoch, later to be used in the training loop.
        """
        self.train()
        running_loss = 0.0
        for batch in loader:
            X, Y = batch          
            Y = Y.float()
            
            self.optimizer.zero_grad()
            outputs = torch.flatten(self(X))
            loss = self.criterion(outputs, Y)        
            loss.backward()        
            self.optimizer.step()
            
            running_loss += loss.item()
        return running_loss / len(loader)
    

    def evaluate(self, loader):
        """
        Evaluate the current model, return the average loss and the accuracy of the model given a set of test inputs.
        """
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                X, Y = batch           
                Y = Y.float()        
            
                outputs = torch.flatten(self(X))
                loss = self.criterion(outputs, Y)
                
                running_loss += loss.item()            
                predicted = (outputs > 0.5).float()  
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
                
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(loader)
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs, filename):
        """
        The loop that traines the model on the training and validation loader for the number of epochs, and saves the best model to a file.
        """
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.num_epochs = num_epochs
            train_loss = self.train_one_opoch(train_loader)
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = best_val_loss
                torch.save(self.state_dict(), filename)
                print(f'New best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')
    
    def predict(self,X):
        """
        use the model to make predictions for new observations.
        """
        output = self.forward(X)
        predictions = []
        for i in range(len(output)):
            if output[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions