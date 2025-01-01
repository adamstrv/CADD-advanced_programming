
import torch.nn as nn
from skorch import NeuralNetClassifier


# this is just a modded model from a website, i'll merge it toghether with my old model later today
 
class NN_BinClass(nn.Module):
    def __init__(self, input_size = 80 , n_neurons = 40, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []

        for i in range(n_layers):
            self.layers.append(nn.Linear(input_size, n_neurons))
            self.acts.append(nn.Sigmoid())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
            input_size = n_neurons
        self.output = nn.Linear(n_neurons, 1)
 
    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x
 
model = NeuralNetClassifier(
    module=NN_BinClass,
    max_epochs=150,
    batch_size=10,
    module__n_layers=3,
)

print(model.initialize())

