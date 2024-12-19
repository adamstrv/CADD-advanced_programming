from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_extraction import extract_train_smiles
from extract_descriptor_types import extract_descriptor_types

train_smiles = extract_train_smiles('train.csv')                # Importing data
descriptor_frame = extract_descriptor_types(train_smiles)

Y = descriptor_frame['classification'].values
X = descriptor_frame.drop('classification', axis = 'columns').values


X = StandardScaler().fit_transform(X)



def dimensionality_reduction(X, variance = 0.95):
    number_components = 1
    explained_variance = []

    while sum(explained_variance) < variance:
        number_components += 1
        
        name_list = []
        for i in range(number_components):
            name_list.append('PC'+ str(i+1))

        pca = PCA(n_components = number_components)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents, columns= name_list)

        explained_variance = pca.explained_variance_ratio_
        print('n_components: ',number_components, 'total variance: ', sum(explained_variance))

    print('we have ', len(explained_variance), ' principal component to cover >95% variance')

    return principalDf