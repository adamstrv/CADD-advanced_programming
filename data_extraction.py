# Function which extracts data from given .csv files

def extract_train_data(train_csv_path):
    """
    Function which extract the train and test data from an input .csv file. It extracts column data
    and creates a dictionary in which data is catagorized based on the characteristic if a SMILES string 
    can or can't bind to the dopamine receptor DRD3 for the training data. It only extract values from the 
    columns for the test data.
    Output: The function returns four things, a list of SMILES strings for the test set, a dictionary with all
    values for the training data, a separate dictionary for only strings that CAN bind, and a separate dictionary 
    of the strings that CAN'T bind with the DRD3 dopamine receptor.
    """

    train_file = open(train_csv_path)    # Open and read file
    train_lines = train_file.readlines()
    train_file.close()

    train_smiles = {}                                       # Make empty dictionary for train SMILES
 
    for line in train_lines[1:]:                            # Iterate through list, except the header  
        values = line.strip().split(',')                    # Remove any whitespace from line and split the line based on comma's.
        smiles = values[0].strip('"')                       # The SMILES string is the first element of the list made of split line
        binary_classification = int(values[1].strip('"'))   # Classification is the second element
        train_smiles[smiles] = binary_classification        # Add to dictionary
    
 #   test_smiles = []                                        # Test data:  make empty list for test data
   

    return train_smiles

r = extract_train_data('train.csv')


#test_file = open(test_csv_path)

# Make list of training values
#test_lines = test_file.readlines()
    
# Iterate thourgh test lines: Skip the first line (header)   
#for line in test_lines[1:]:  
#    # The SMILES string is the second element of the list made of split line
#    smiles = line.strip().split(',')[1]
#    cleaned_smiles = smiles.strip('"')
#    test_smiles.append(cleaned_smiles)

#test_smiles = []  