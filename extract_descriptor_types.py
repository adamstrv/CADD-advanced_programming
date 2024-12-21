from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
import pandas as pd

from data_extraction import extract_test_smiles

def train_descriptor_frame(smiles_strings):
    """
    Function that extract ALL descriptor types from an input SMILES strings and returns these.
    It makes two cases for two different inputs: 
    If list and if dictionary (assuming the keys of the dictionary contain the SMILES string)
    Output:
    Returns an excel sheet of the dataframe containing the values of the descriptor types for all molecules. 
    !!! The index represents the 'Unique_ID' of the SMILES strings !!!
    """
    
    descriptors = []                                                 # Make empty list for descriptors
 
    if isinstance(smiles_strings, dict):                             # Make case for training set: dictionary input
        for smiles in smiles_strings.keys():                         # Iterate over SMILES in dictionary and make list of descriptor types
            mol = Chem.MolFromSmiles(smiles)                         # Convert SMILES string into RDKit molecule object
            descriptor = Descriptors.CalcMolDescriptors(mol)         # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
        data_frame = pd.DataFrame(descriptors) 
        data_frame['classification'] = list(smiles_strings.values())
    
    else:
        raise TypeError("Input has to be either a list or dictionary (with the keys being the SMILES string).")

    return data_frame



def test_descriptor_frame(smiles_strings):
    """
    Function that extract ALL descriptor types from an input SMILES strings and returns these.
    It makes two cases for two different inputs: 
    If list and if dictionary (assuming the keys of the dictionary contain the SMILES string)
    Output:
    Returns an excel sheet of the dataframe containing the values of the descriptor types for all molecules. 
    !!! The index represents the 'Unique_ID' of the SMILES strings !!!
    """
    
    descriptors = []                                                 # Make empty list for descriptors
 
    if isinstance(smiles_strings, dict):                             # Make case for training set: dictionary input
        for smiles in smiles_strings.values():                         # Iterate over SMILES in dictionary and make list of descriptor types
            mol = Chem.MolFromSmiles(smiles)                         # Convert SMILES string into RDKit molecule object
            descriptor = Descriptors.CalcMolDescriptors(mol)         # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
        data_frame = pd.DataFrame(descriptors) 
        data_frame['ID'] = list(smiles_strings.keys())
    
    else:
        raise TypeError("Input has to be either a list or dictionary (with the keys being the SMILES string).")

    return data_frame