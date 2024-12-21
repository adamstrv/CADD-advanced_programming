import torch
from neural_network import NN_BinClass

trained_model = NN_BinClass(80)
trained_model.load_state_dict(torch.load('best_model.pth',weights_only=True))