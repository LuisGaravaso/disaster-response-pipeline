# Data
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from collections import defaultdict
import pickle
from warnings import filterwarnings

#Sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# NLP 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download(['punkt','stopwords','wordnet'])
filterwarnings('ignore')

def load_data(database_filepath):
    
    #Read data from Database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_data',engine)
    
    # Break data into X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis = 1)
    Y = Y.astype(np.uint8)
    
    # Some values in the 'related' column are
    # marked as 2 where they should be 0
    # The next two lines fix this problem
    bol = Y['related'] == 2
    Y.loc[bol,'related'] = 0
    
    #Return 
    return X, Y, Y.columns

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
    steps["count"] = CountVectorizer(tokenizer = tokenize) 
    steps["tfidf"] = TfidfTransformer(norm='l1')
    
    ranforest = RandomForestClassifier(n_estimators=50,
                                       criterion='gini') #RandomForest
    clf = MultiOutputClassifier(ranforest, n_jobs = -1) #Make MultiOutput CLF
    steps["Classifier"] = clf #Add classifier to the end of the Pipeline
    steps = list(steps.items()) #Convert Steps to list
    
    pipeline = Pipeline(steps) #Make Pipeline 

    return pipeline     

def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test) #Predict
    
    #List to save Evaluation Metrics' Results
    Acc = [] #Accuracy
    Prc = [] #Precision
    Rec = [] #Recall
    
    #Evaluate every column
    for ind, col in enumerate(Y_test.columns):

        y_true = Y_test[col]
        y_pred = Y_pred[:,ind] 
        
        #Metrics 
        acc = accuracy_score(y_true, y_pred) #Accuracy
        prc = precision_score(y_true, y_pred) #Precision
        rec = recall_score(y_true, y_pred) #Recall
        
        Acc.append(acc)
        Prc.append(prc)
        Rec.append(rec)
    
    #Create dataset to save evaluation results into a .csv file
    data = np.c_[Acc, Prc, Rec]
    Eval = pd.DataFrame(data, index = category_names,
                        columns = ['Accuracy','Precision','Recall'])
    Eval.to_csv('evaluation_results.csv')

def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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