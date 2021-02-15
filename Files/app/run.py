import json
import plotly
import pandas as pd
import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
from sqlalchemy import create_engine

from collections import Counter

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model
with open("../models/classifier.pkl", 'rb') as model: 
    model = pickle.load(model)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df['genre'].value_counts()
    genre_names = list(genre_counts.index)
    
    #most frequent words
    text = " ".join(df["message"]) #Make a big text string
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) #Remove punctuation
    
    words = text.split(" ") #List of words
    #Remove Stop Words
    stop = stopwords.words('english') 
    words = [word for word in words if (word not in stop) and (word != "")]
    WordCount = Counter(words).most_common(20) #20 most common words
    word_labels, word_counts = list(zip(*WordCount)) #labels and counts
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=word_counts,
                    y=word_labels,
                    orientation='h',
                    marker=dict(color = word_counts,
                                colorscale='viridis'),
                )
            ],

            'layout': {
                'title': '20 Most Frequent Words',
                'yaxis': {
                    'title': "Word"
                },
                'xaxis': {
                    'title': "Count"
                },
                'height':600,
                'width':1200
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(df.columns)
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()