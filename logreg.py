import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot
import text

from pymongo import MongoClient
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

COLLECTION_NAME = 'clean_df'
DB_NAME = 'bda_project'


def run():
    mongoClient = MongoClient()

    db = mongoClient.get_database(DB_NAME)
    collection = db[COLLECTION_NAME]
    data = collection.find()
    df = pd.DataFrame(data)
    df.pop('_id')

    # Add columns with Subjectivity and Polarity values for the daily news sentiment
    df['Subjectivity'] = df['News'].apply(text.get_subjectivity)
    df['Polarity'] = df['News'].apply(text.get_polarity)

    # Get Sentiment score for each day:
    compound = []
    neg = []
    pos = []
    neu = []
    SIA = 0

    for i in range(0, len(df['News'])):
        SIA = text.get_sia(df['News'][i])
        compound.append(SIA['compound'])
        neg.append(SIA['neg'])
        pos.append(SIA['pos'])
        neu.append(SIA['neu'])

    # Add the previous Sentiment analysis values to the dataset
    df['compound'] = compound
    df['neg'] = neg
    df['pos'] = pos
    df['neu'] = neu

    ### DATA PREP AND MODELS FOR DJIA_LABEL ###
    # Create the feature data set:
    X = df
    X = np.array(X.drop(['Date', 'DJIA_LABEL', 'DJIA_CLOSE', 'OIL_LABEL', 'OIL_CLOSE', 'News'], axis=1))
    print('X:')
    print(X)
    # Create the target data set:
    y = np.array(df['DJIA_LABEL'])
    print('Y:')
    print(y)
    # Split the data:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ### LOGISTIC REGRESSION ### DJIA_LABEL ###
    # Import Logistic Regression model:
    # Create and train the Logistic Regression model:
    model1 = LogisticRegression().fit(x_train, y_train)
    # Model predicton:
    predict1 = model1.predict(x_test)
    print('Prediction:')
    print(predict1)
    # Print the classification report:
    print(classification_report(y_test, predict1))
    # Import and plot confusion matrix for the result:
    scikitplot.metrics.plot_confusion_matrix(y_test, predict1)
    plt.show()

    ### LINEAR DISCRIMINANT ANALYSIS ### DJIA_LABEL ###
    # Import Linear Discriminant Analysis model:
    # Create and train the Linear Discriminant Analysis model:
    model2 = LinearDiscriminantAnalysis().fit(x_train, y_train)
    # Model predicton:
    predict2 = model2.predict(x_test)
    print(predict2)
    # Print the classification report:
    print(classification_report(y_test, predict2))
    # Import and plot confusion matrix for the result:
    scikitplot.metrics.plot_confusion_matrix(y_test, predict2)
    plt.show()

    ### DATA PREP AND MODELS FOR OIL_LABEL ###
    # Create the feature data set:
    X = df
    X['OIL_CLOSE'] = X.OIL_CLOSE.replace(np.nan, 0)
    X = np.array(X.drop(['Date', 'DJIA_LABEL', 'DJIA_CLOSE', 'OIL_LABEL', 'OIL_CLOSE', 'News'], axis=1))
    print(X)
    # create the target data set:
    y = np.array(df['OIL_LABEL'])
    print(y)
    # Split the data:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ### LOGISTIC REGRESSION ### OIL_LABEL ###
    # Create and train the Logistic Regression model:
    model3 = LogisticRegression().fit(x_train, y_train)
    # Model predicton:
    predict3 = model3.predict(x_test)
    print(predict3)
    # Print the classification report:
    print(classification_report(y_test, predict3, zero_division=0))
    # Plot confusion matrix for the result:
    scikitplot.metrics.plot_confusion_matrix(y_test, predict3)
    plt.show()

    ### LINEAR DISCRIMINANT ANALYSIS ### OIL_LABEL ###

    # Create and train the Linear Discriminant Analysis model:
    model4 = LinearDiscriminantAnalysis().fit(x_train, y_train)
    # Model predicton:
    predict4 = model4.predict(x_test)
    print(predict4)
    # Print the classification report:
    print(classification_report(y_test, predict4, zero_division=0))
    # Plot confusion matrix for the result:
    scikitplot.metrics.plot_confusion_matrix(y_test, predict4)
    plt.show()

    # Linear Discriminant Analysis model for OIL_LABEL - using ONLY the 'compound' value of the daily sentiment
    # Create the feature data set:
    X = df
    X['OIL_CLOSE'] = X.OIL_CLOSE.replace(np.nan, 0)
    X = np.array(X.drop(
        ['Date', 'DJIA_LABEL', 'DJIA_CLOSE', 'OIL_LABEL', 'OIL_CLOSE', 'News', 'Subjectivity', 'Polarity', 'neg', 'pos',
         'neu'], axis=1))
    print(X)
    # Create the target data set:
    y = np.array(df['OIL_LABEL'])
    print(y)
    # Split the data:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create and train the model:
    model = LinearDiscriminantAnalysis().fit(x_train, y_train)
    # Model predicton:
    predict = model.predict(x_test)
    print(classification_report(y_test, predict, zero_division=0))

    scikitplot.metrics.plot_confusion_matrix(y_test, predict)
    plt.show()
