from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


class custom_PCA(PCA):
    def __init__(self, X, variance = 0.95):
        number_components = 1
        explained_variance = []

        while sum(explained_variance) < variance:
            number_components += 1
            
            self.name_list = []
            for i in range(number_components):
                self.name_list.append('PC'+ str(i+1))

            self.pca = PCA(n_components = number_components)
            principalComponents = self.pca.fit(X)

            explained_variance = self.pca.explained_variance_ratio_
        
    def transform_frame(self, X):
        principalComponents = self.pca.transform(X)
        principalDf = pd.DataFrame(data = principalComponents, columns= self.name_list)

        return principalDf