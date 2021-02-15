import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    This function converts the .csv files into a Pandas DataFrame

    Parameters
    ----------
    messages_filepath : str
        path to the .csv file containing the messages
    categories_filepath : str
        path to the .csv file containing the categories

    Returns
    -------
    pandas DataFrame
        returns the two .csv files merges on the 'id' column
    """
    
    #Load Datasets
    messages = pd.read_csv(messages_filepath,
                           usecols = ['id','message','genre'])
    categories = pd.read_csv(categories_filepath, sep = ",")
    
    #Return merged Dataset
    return pd.merge(messages, categories)
    
def clean_data(df):
    
    """
    This function preprocesses the DataFrame obtained from load_data().
    It will break the categories column into 36 individual target columns.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame obtained from load_data()

    Returns
    -------
    pandas DataFrame
        returns another DataFrame with 36 individual target columns.

    """
    
    # Create a dataframe of the 36 individual category columns
    cat_index = df["id"] #index
    messages = df.drop(['categories'], axis = 1) #message
    cat_data = df["categories"].str.split(";", expand = True) #data
    
    #Since Data looks like:
    # related-1, request-0, offer-0, aid_related-0, medical_help-0, ...
    #The next lines split the data into labels and values
    #The labels will then be converted into column names
    row = cat_data.iloc[0] 
    cat_colnames = row.str.split("-", expand = True)[0]
    cat_data.columns = cat_colnames
    
    for column in cat_colnames:
        # set each value to be the last character of the string
        cat_data[column] = cat_data[column].str.split("-",expand = True)[1]
    
    #Convert from str to integer
    cat_data = cat_data.astype(np.uint8)
    
    # Some values in the 'related' column are
    # marked as 2 where they should be 0
    # The next two lines fix this problem
    bol = cat_data['related'] == 2
    cat_data.loc[bol,'related'] = 0
    
    #Recreate 'categories' DataFrame with proper index 
    categories = cat_data.set_index(cat_index).reset_index()

    #Return merged Dataset
    return pd.merge(messages, categories).drop_duplicates()

def save_data(df, database_filename):
    """
    This function saves a pandas DataFrame into a SQL database.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to be saved into the SQL database
    database_filename : str
         Path to SQL Database

    Returns
    -------
    None.

    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_data', engine, index=False, if_exists = 'replace')  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()