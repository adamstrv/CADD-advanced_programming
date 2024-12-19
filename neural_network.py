import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from sklearn.preprocessing import StandardScaler

from data_extraction import extract_train_smiles
from extract_descriptor_types import extract_descriptor_types
from PCA import custom_pca_fit


train_smiles = extract_train_smiles('train.csv')                # Importing data
train_descriptor_frame = extract_descriptor_types(train_smiles)

Y_train = train_descriptor_frame['classification'].to_numpy()
X_train = train_descriptor_frame.drop('classification', axis = 'columns').to_numpy()
X_train = StandardScaler().fit_transform(X_train)


reduced_X_train, pca = custom_pca_fit(X_train)

print(reduced_X_train.shape)




#model = Sequential([
 #   Dense(units=16, activation='relu', input_shape=(num_features,)),
  #  Dense(units=32, activation='relu'),
  #  Dense(units=1, activation='sigmoid')
#])
#
#model.compile(optimizer='adam',
##              loss='binary_crossentropy',
#              metrics=['accuracy'])#
#
