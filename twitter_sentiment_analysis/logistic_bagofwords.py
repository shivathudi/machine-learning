import pandas as pd
import sys
from tweet_processing import tokenize_tweet

import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Reading and Processing Data
train_file = sys.argv[1]
test_file = sys.argv[2]

df_columns = ['sentiment', 'text']

df = pd.read_csv(train_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns,encoding='ISO-8859-1')
test_df = pd.read_csv(test_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns,encoding='ISO-8859-1')

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = tokenize_tweet(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems


vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = False,
    stop_words = None,
    ngram_range= (1,2),
)

# Build the corpus

corpus_data_features = vectorizer.fit_transform(
    df.text.tolist() + test_df.text.tolist())

corpus_data_features_nd = corpus_data_features.toarray()

# Add the function word rates

function_words = ['i', 'the', 'and', 'to', 'a', 'of', 'that', 'in',
                  'it', 'my', 'is', 'you', 'was', 'for', 'have', 'with', 'he', 'me',
                  'on', 'but']

punct_words = ['.', ',', '!']

train_size = df.size / 2
test_size = test_df.size / 2
features = vectorizer.get_feature_names()

my_array = np.zeros( (train_size +test_size, len(features) + len(function_words) + len(punct_words)) )

for i in range(0,train_size):

    text = df.loc[i,'text']
    text.replace('.', '. ')
    tokenized_list = nltk.word_tokenize(text)
    counts = Counter(tokenized_list)

    row_vector = []
    for word in function_words + punct_words:
        numerator = 0.0
        denominator = float(len(tokenized_list))
        numerator += counts[word]
        row_vector.append(numerator/denominator)

    row_vector = np.array(row_vector)

    my_array[i] = np.append(corpus_data_features_nd[i], row_vector)

for i in range(0,test_size):

    text = test_df.loc[i,'text']
    text.replace('.', '. ')
    tokenized_list = nltk.word_tokenize(text)
    counts = Counter(tokenized_list)

    row_vector = []
    for word in function_words + punct_words:
        numerator = 0.0
        denominator = float(len(tokenized_list))
        numerator += counts[word]
        row_vector.append(numerator/denominator)

    row_vector = np.array(row_vector)

    my_array[i+train_size] = np.append(corpus_data_features_nd[i+train_size], row_vector)


# Fit the model and check performance on testing set

X_train = my_array[0:len(df)]
Y_train = df.sentiment

X_test = my_array[len(df):]
Y_test = test_df.sentiment

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=Y_train)

y_pred = log_model.predict(X_test)

print "---------------------------------------------------------------"
print "MISCLASSIFICATION RATE = ", (1 -log_model.score(X_test, Y_test))

print "---------------------------------------------------------------"
print(classification_report(Y_test, y_pred))
