import torch
from torch import nn
import matplotlib.pyplot as plt

class NN_BinClass(nn.Module):
    def __init__(self, input_size, learning_rate = 0.005, momentum = 0.95):
        super(NN_BinClass, self).__init__()

        self.linear1 = nn.Linear(input_size, round(input_size/2))
        self.sigmoid1 = nn.Sigmoid() 
        self.linear2 = nn.Linear(round(input_size/2), round(input_size/4))        
        self.sigmoid2 = nn.Sigmoid()       
        self.output = nn.Linear(round(input_size/4), 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.4)

        self.lr = learning_rate
        self.momentum = momentum

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), self.lr , self.momentum)

    def forward(self, x):   
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.dropout(x)
        x = self.output(x)
        out = self.sigmoid(x)
        
        return out

    def train_one_opoch(self, loader, learning_rate= 0.05, momentum = 0.95):
        self.lr = learning_rate
        self.momentum = momentum
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
    
    def training_loop(self, train_loader, val_loader, num_epochs):
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
                torch.save(self.state_dict(), 'best_model.pth')
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