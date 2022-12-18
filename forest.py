import math
import re
import string
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


def run():
    mongoClient = MongoClient()

    db = mongoClient.clean_df
    data = db.segment.find()
    df = pd.DataFrame(data)
    df.pop('_id')
    print(df.head())

    # Dictionary of English Contractions
    contractions_dict = {"ain't": "are not", "it's": "it is", "aren't": "are not",
                         "can't": "cannot", "can't've": "cannot have",
                         "'cause": "because", "could've": "could have", "couldn't": "could not",
                         "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                         "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                         "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                         "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                         "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                         "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                         "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
                         "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                         "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                         "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                         "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                         "mustn't've": "must not have", "needn't": "need not",
                         "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                         "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                         "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                         "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                         "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                         "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                         "there'd've": "there would have", "they'd": "they would",
                         "they'd've": "they would have", "they'll": "they will",
                         "they'll've": "they will have", "they're": "they are", "they've": "they have",
                         "to've": "to have", "wasn't": "was not", "we'd": "we would",
                         "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                         "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                         "what'll've": "what will have", "what're": "what are", "what've": "what have",
                         "when've": "when have", "where'd": "where did", "where've": "where have",
                         "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                         "why've": "why have", "will've": "will have", "won't": "will not",
                         "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                         "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                         "y'all'd've": "you all would have", "y'all're": "you all are",
                         "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                         "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                         "you've": "you have"}

    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    # Function for expanding contractions
    def expand_contractions(text,contractions_dict):
      def replace(match):
        return contractions_dict[match.group(0)]
      return contractions_re.sub(replace, text)
    # Expanding Contractions in the reviews
    df['News']=df['News'].apply(lambda x:expand_contractions(x, contractions_dict))
    for index,text in enumerate(df['News'][35:37]):
      print('News %d:\n'%(index+1),text)

    # Lowercase the news
    df['lemmatized'] = df['News'].apply(lambda x: x.lower())
    # Remove digits and words including digits
    df['lemmatized'] = df['lemmatized'].apply(lambda x: re.sub('\w*\d\w*', '', x))
    # Remove punctuation
    df['lemmatized'] = df['lemmatized'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    # Removing extra spaces
    df['lemmatized'] = df['lemmatized'].apply(lambda x: re.sub(' +', ' ', x))
    for index, text in enumerate(df['lemmatized'][35:37]):
        print('News %d:\n' % (index + 1), text)

    # Loading model
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Lemmatization with stopwords removal
    df['lemmatized'] = df['lemmatized'].apply(
        lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop == False)]))
    for index, text in enumerate(df['lemmatized'][35:37]):
        print('News %d:\n' % (index + 1), text)

    # Creating Document Term Matrix
    cv = CountVectorizer(analyzer='word')
    data = cv.fit_transform(df['lemmatized'])
    df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names_out())
    df_dtm.index = df.index
    print(df_dtm.head(3))

    # Function for generating word clouds
    def generate_wordcloud(data, title):
        wc = WordCloud(width=400, height=330, max_words=150, colormap="Dark2").generate_from_frequencies(data)
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title('\n'.join(wrap(title, 60)), fontsize=13)
        plt.show()
    # Transposing document term matrix
    df_dtm = df_dtm.transpose()
    # Plotting word cloud for top5 compound sentiment using indices for top5/bottom5 sentiment rows (also listed down below)
    # Wordcloud visualization of most common words in news for top5 positive compound sentiment
    generate_wordcloud(df_dtm[1567].sort_values(ascending=False), "Top1 positive compound sentiment")
    generate_wordcloud(df_dtm[1558].sort_values(ascending=False), "Top2 positive compound sentiment")
    generate_wordcloud(df_dtm[785].sort_values(ascending=False), "Top3 positive compound sentiment")
    generate_wordcloud(df_dtm[489].sort_values(ascending=False), "Top4 positive compound sentiment")
    generate_wordcloud(df_dtm[1240].sort_values(ascending=False), "Top5 positive compound sentiment")