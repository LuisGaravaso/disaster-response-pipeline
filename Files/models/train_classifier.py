# Data
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from collections import defaultdict

#Sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# NLP 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def load_data(database_filepath):
    
    #Read data from Database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_data',engine)
    
    # Break data into X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message'], axis = 1)
    
    # Some values in the 'related' column are
    # marked as 2 where they should be 0
    # The next two lines fix this problem
    bol = Y['related'] == 2
    Y.loc[bol,'related'] = 0
    
    #Return 
    return X, Y.astype(np.uint8), Y.columns

def tokenize(text):
    
    #Get StopWords and Lemmatizer 
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove Stop Words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    #Steps of the Pipeline
    steps = defaultdict(list)
    steps["count"] = CountVectorizer(tokenizer = tokenize) #
    steps["tfidf"] = TfidfTransformer(norm='l2')
    
    ranforest = RandomForestClassifier(n_estimators='50',
                                       criterion='gini') #RandomForest
    clf = MultiOutputClassifier(ranforest, n_jobs = -1) #Make MultiOutput CLF
    steps["Classifier"] = clf #Add classifier to the end of the Pipeline
    steps = list(steps.items()) #Convert Steps to list
    
    pipeline = Pipeline(steps) #Make Pipeline 

    return pipeline     

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()