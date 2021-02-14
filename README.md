# disaster-response-pipeline
This repository is part of Udacity's Data Science Nanodegree Course. \
Its goal is too use ETL and NLP techniques to construct a Disaster Response Pipeline.

## Table of contents
* [Technologies]($Technologies)
* [Installation](#Installation)
* [Files](#Files)
* [Usage](#Usage)
* [Future Work](#Future-Work)

## Technologies

This project uses:

* Python 3.8.7

And the packages:

* Sys: built-in
* Pandas: 1.2.1
* Numpy: 1.19.2
* Sqlalchemy: 1.3.21
* Re: 2.2.1
* Collections: built-in
* Pickle: built-in
* Warnings: built-in
* Sklearn: 0.23.2
* Nltk: 3.5
* Json: 2.0.9
* Plotly: 4.14.3
* Flask: 1.0.3

## Installation

```
$ git clone https://github.com/LuisGaravaso/disaster-response-pipeline
```

## Files

Once you clone the repo, you'll get the following items:

* `LICENSE`: MIT License
* `README`: This README
* `Files`: Folder containing the project files

The `Files` folder contains the following `folders`:

* `data`: contains the data and the `etl` file
* `models`: contains the `train_classifier.py` file, responsible for training, evaluating and saving a *Machine Learning* model. \
  **NOTE**: I used a `RandomForestClassifier` from `scikit-learn`, but feel free to try another one.

* `app`: contains the `Flask` application and the `.html` files responsible for generating the web pages.

## Usage

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**NOTE**: These instructions were given by Udacity

## Future Work

I do not have intentions of continuing this work.
