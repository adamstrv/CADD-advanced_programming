# Extract descriptor types for SMILES strings

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

#rdkit.Chem.Descriptors.CalcMolDescriptors -> dictionary descriptor: value

# Import function which extracts data from .csv files
from data_extraction import extract_data

def extract_descriptor_types(smiles_strings):
    """
    Function which extract ALL descriptor types from an input SMILES strings and returns these.
    It makes two cases for two different inputs: 
    If list and if dictionary (assuming the keys of the dictionary contain the SMILES string)
    Output:
    Returns an excel sheet of the dataframe containing the values of the descriptor types for all molecules. 
    !!! The index represents the 'Unique_ID' of the SMILES strings !!!
    """
    # Make empty list for descriptors
    descriptors = []
      
    # Make case for test set: it is just a list input
    if isinstance(smiles_strings, list):
        for smiles in smiles_strings:
            # Convert SMILES string into RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            descriptor = Descriptors.CalcMolDescriptors(mol) 
            # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
    
    # Make case for training set: dictionary input
    elif isinstance(smiles_strings, dict):
        # Iterate over SMILES in dictionary and make list of descriptor types
        for smiles in smiles_strings.keys():
            # Convert SMILES string into RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            descriptor = Descriptors.CalcMolDescriptors(mol)
            # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
    else:
        raise TypeError("Input has to be either a list or dictionary (with the keys being the SMILES string).")
    
    # Create dataframe to return results
    df = pd.DataFrame(descriptors)
    
    # Adjust index to start from 1 since index represents 'Unique_ID'
    df.index = pd.RangeIndex(start = 1, stop = len(df) + 1)
    
    # Return results in excel file
    df.to_excel('Descriptor_analysis.xlsx', sheet_name='descriptor_results', index=True)
    return descriptors



test_smiles, train_smiles, train_can_bind, train_cant_bind = extract_data('Book1.csv', 'train.csv')
extract_descriptor_types(test_smiles)