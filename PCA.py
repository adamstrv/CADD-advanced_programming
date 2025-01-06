from sklearn.decomposition import PCA
import pandas as pd

class custom_PCA(PCA):
    """
    Basically a pca class, but you can choose how much of the variance you (minimally) want to capture with your dimentionality reduction
    """
    def __init__(self, X, variance = 0.95):
        """
        As long as the variance is lower than the variance we minimally want, it runs a pca an checks the explained variance,
        each time increasing the number of principal components untill the minimal wanted variance is met.
        """
 
        number_components = 1
        explained_variance = [],


        while sum(explained_variance) < variance:
            number_components += 1
            
            self.name_list = []
            for i in range(number_components):
                self.name_list.append('PC'+ str(i+1))

            self.pca = PCA(n_components = number_components)
            self.pca.fit(X)

            explained_variance = self.pca.explained_variance_ratio_
        
    def transform_frame(self, X):
        """
        Transform the data X to the new pca, and return it as pandas dataframe.
        """
        principalComponents = self.pca.transform(X)
        principalDf = pd.DataFrame(data = principalComponents, columns= self.name_list)
        return principalDf 