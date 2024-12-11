# Function which extracts data from given .csv files

def extract_data(test_csv_path, train_csv_path):
    """
    Function which extract the train and test data from an input .csv file. It extracts column data
    and creates a dictionary in which data is catagorized based on the characteristic if a SMILES string 
    can or can't bind to the dopamine receptor DRD3 for the training data. It only extract values from the 
    columns for the test data.
    Output: The function returns four things, a list of SMILES strings for the test set, a dictionary with all
    values for the training data, a separate dictionary for only strings that CAN bind, and a separate dictionary 
    of the strings that CAN'T bind with the DRD3 dopamine receptor.
    """
    
    # Make empty dictionary for train SMILES
    train_smiles = {}
    # Make dictionaries for SMILES strings that can and 
    # can't bind to DRD3
    train_can_bind = {}
    train_cant_bind = {}
    

    # Open files
    test_file = open(test_csv_path, 'r')
    train_file = open(train_csv_path, 'r')  
    # Train data: Make list of training values
    train_lines = train_file.readlines()
    
    # Iterate through list, except the header   
    for line in train_lines[1:]:
        # Remove any whitespace from line and split the line based on comma's.
        values = line.strip().split(',')
        # The SMILES string is the first element of the list made of split line
        smiles = values[0]
        # Classification is the second element
        binary_classification = int(values[1].strip('"'))
        
        # Add to dictionary
        train_smiles[smiles] = binary_classification
    
        # Add to appropriate dictionaries
        if binary_classification == 1:
            train_can_bind[smiles] = binary_classification
        elif binary_classification == 0:
            train_cant_bind[smiles] = binary_classification
            
    # Test data:  make empty list for test data
    test_smiles = []
   
    # Make list of training values
    test_lines = test_file.readlines()
     
    # Iterate thourgh test lines: Skip the first line (header)   
    for line in test_lines[1:]:  
        # The SMILES string is the second element of the list made of split line
        smiles = line.strip().split(',')[1]
        test_smiles.append(smiles)
        
    return test_smiles, train_smiles, train_can_bind, train_cant_bind