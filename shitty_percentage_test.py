import torch

from data_extraction import extract_train_smiles
from extract_descriptor_types import train_descriptor_frame

from train_network import scaler
from train_network import pca
from final_model import trained_model

train_smiles = extract_train_smiles('train.csv')                              #   Extract a library with data from the a csv file
train_frame = train_descriptor_frame(train_smiles)                            #   Extract a dataframe with features of each smile via RDkit
Y = train_frame['classification'].tolist()                                    #   Define Y with the classification
X = train_frame.drop('classification', axis = 'columns')                      #   Define X with all the features

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