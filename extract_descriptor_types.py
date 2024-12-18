# Extract descriptor types for SMILES strings
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

#rdkit.Chem.Descriptors.CalcMolDescriptors -> dictionary descriptor: value

# Import function which extracts data from .csv files
from data_extraction import extract_train_smiles

def extract_descriptor_types(smiles_strings):
    """
    Function which extract ALL descriptor types from an input SMILES strings and returns these.
    It makes two cases for two different inputs: 
    If list and if dictionary (assuming the keys of the dictionary contain the SMILES string)
    Output:
    Returns an excel sheet of the dataframe containing the values of the descriptor types for all molecules. 
    !!! The index represents the 'Unique_ID' of the SMILES strings !!!
    """
    descriptors = []                                                 # Make empty list for descriptors
    
    if isinstance(smiles_strings, list):                             # Make case for test set: it is just a list input
        for smile in smiles_strings:
            mol = Chem.MolFromSmiles(smile)                         # Convert SMILES string into RDKit molecule object
            descriptor = Descriptors.CalcMolDescriptors(mol)         # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
        data_frame = pd.DataFrame(descriptors)  

    elif isinstance(smiles_strings, dict):                           # Make case for training set: dictionary input
        for smiles in smiles_strings.keys():                         # Iterate over SMILES in dictionary and make list of descriptor types
            mol = Chem.MolFromSmiles(smiles)                         # Convert SMILES string into RDKit molecule object
            descriptor = Descriptors.CalcMolDescriptors(mol)         # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
        data_frame = pd.DataFrame(descriptors) 
        data_frame['classification'] = list(smiles_strings.values())
    
    else:
        raise TypeError("Input has to be either a list or dictionary (with the keys being the SMILES string).")
    
    # Create dataframe to return results

    # df.index = pd.RangeIndex(start = 1, stop = len(df) + 1)                                  # Adjust index to start from 1 since index represents 'Unique_ID'
    # df.to_excel('Descriptor_analysis.xlsx', sheet_name='descriptor_results', index=True)     # Return results in excel file
    return data_frame


train_smiles= extract_train_smiles('shortertrain.csv')
predictors = extract_descriptor_types(train_smiles)
print(predictors)