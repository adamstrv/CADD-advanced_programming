from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def extract_train_smiles(train_csv_path):
    """
    Function that extract the smiles with their classification from a .csv file
    it returns a dictonary with a smile strings and their classifications
    """

    train_file = open(train_csv_path)                       # Open and read file
    train_lines = train_file.readlines()
    train_file.close()

    train_smiles = {}                                       # Make empty dictionary for train SMILES
 
    for line in train_lines[1:]:                            # Iterate through list, except the header  
        values = line.strip().split(',')                    # Remove any whitespace from line and split the line based on comma's.
        smiles = values[0].strip('"')                       # The SMILES string is the first element of the list made of split line
        binary_classification = int(values[1].strip('"'))   # Classification is the second element
        train_smiles[smiles] = binary_classification        # Add to dictionary

    return train_smiles


def extract_test_smiles(test_csv_path):
    """
    Function that extract the smiles with their classification from a .csv file
    it returns a list with a smile strings
    """

    test_file = open(test_csv_path)                         # Open and read file
    test_lines = test_file.readlines()
    test_file.close()

    test_smiles = {}                                        # Make empty list for train SMILES
    
    for line in test_lines[1:]:                             # Iterate thourgh test lines: Skip the first line (header) 
        ID, smile = line.strip().split(',')
        cleaned_smile = smile.strip('"')
        cleaned_ID = ID.strip('"')
        test_smiles[cleaned_ID] = cleaned_smile

    return test_smiles


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
        for smiles in smiles_strings.values():                       # Iterate over SMILES in dictionary and make list of descriptor types
            mol = Chem.MolFromSmiles(smiles)                         # Convert SMILES string into RDKit molecule object
            descriptor = Descriptors.CalcMolDescriptors(mol)         # Add to descriptor list, includes dictionary for each molecule
            descriptors.append(descriptor)
        
        data_frame = pd.DataFrame(descriptors) 
        data_frame['ID'] = list(smiles_strings.keys())
    
    else:
        raise TypeError("Input has to be either a list or dictionary (with the keys being the SMILES string).")

    return data_frame


if __name__ == "__main__":
    train_smiles = extract_train_smiles('train.csv')
    train_frame = train_descriptor_frame(train_smiles)
    train_frame.to_csv('training_dataframe.csv')

    test_smiles = extract_test_smiles('test.csv')
    test_frame = test_descriptor_frame(test_smiles)
    test_frame.to_csv('testing_dataframe.csv')