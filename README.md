# Twitter Sentiment Analysis

The Goal of this Experiment was to predict the test data provided, while also providing an open ended research report.

The predicted results are submitted onto Kaggle and compared against the whole class.

## Data
Train, test and development sets are provided.

Each row in the data files contains the sentiment, tweet ID, and the tweet representation as csv.

### Class Labels
The class labels are represented by positive, negative, and neutral sentiments:

[pos, neg, neu]

### Features
The full data will be used for this analysis.

The raw tweets represented as a single string,

e.g. pos,tweetID, “im feeling so happy today, so very happy”

## Sentiment Analysis Process
The following work was conducted to predict the sentiment of the test data
1. Load Tweets
2. Pre-Process the tweets via custom techniques
3. Implement a baseline model
4. Implement two ML models and compare results
5. Choosing the best pre-process techniques, predict the sentiment of test data

The machine learning models used for this analysis are Naive Bayes, Logistic Regression and SGD Classifier

## Final Results
It was shown that using the training data to train the LR and SGD models accuracy score peaked  at 0.795 and 0.797 respectively.

By using both ml models trained from training data, the test labels/sentiment were predicted and submited to Kaggle.

The submission accuracy scored, LR: 0.81015, SGD: 0.80680

# How to Run

## Requirements
The following python libraries are required to run the analysis
* nltk
* spacy
* ftfy
* sklearn
* notebook
* pandas
* numpy

The command below will install the required libraries
```bash
pip install nltk, spacy, ftfy, sklearn, notebook, pandas
py -m spacy download en_core_web_sm
```

## Running the Notebook
There is a provided batch file that will start jupyter notebook, otherwise by using VS code, an notebook extension can be installed

```bash
.\load_notebook.bat
```