# Function which extracts data from given .csv files
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
