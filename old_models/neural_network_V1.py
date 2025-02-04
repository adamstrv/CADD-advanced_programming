import torch
from torch import nn
import matplotlib.pyplot as plt

class NN_BinClass(nn.Module):
    """
    class representing a neural network used for binary classification. When defining the network, you define the
    input size, the amount of neurons in the first layer, the learning rate, and the momentum. The model is structured
    like a funnel: every layer, the amount of neurons half.
    """
    def __init__(self, input_size, first_layer_size, dropout = 0.3, learning_rate = 0.005, momentum = 0.95):
        """
        Define the model and set the input size, size of the first layer, the amount of dropout, the learning rate, and the momentum
        """
        super(NN_BinClass, self).__init__()

        self.linear1 = nn.Linear(input_size, round(first_layer_size))
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(round(first_layer_size), round(first_layer_size/2))
        self.sigmoid2 = nn.Sigmoid()     
        self.output = nn.Linear(round(first_layer_size/2), 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

        self.lr = learning_rate
        self.momentum = momentum

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), self.lr , self.momentum)


    def forward(self, x):
        """
        This runs the model in a straightforward way, calculating an output for a certain input array.
        """
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.sigmoid2(x)

        x = self.dropout(x)
        x = self.output(x)
        out = self.sigmoid(x)
        
        return out

    def train_one_opoch(self, loader):
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
                # apply threshold to get binary predictions
                predicted = (outputs > 0.5).float()  
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
                
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(loader)
        return avg_loss, accuracy
    
    def training_loop(self, train_loader, val_loader, num_epochs, filename):
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
            
            # Check for the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = best_val_loss
                torch.save(self.state_dict(), filename)
                print(f'New best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')
    
    def show_loss_curves(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.num_epochs+1), self.train_losses, label='Training Loss')
        plt.plot(range(1, self.num_epochs+1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.show()
    
    def predict(self,X):
        output = self.forward(X)
        predictions = []
        for i in range(len(output)):
            if output[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions