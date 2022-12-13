from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Subjectivity:
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Polarity:
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Create a function to get the sentiment score:
def get_sia(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment