import pandas as pd
import sys
from tweet_processing_original import tokenize_tweet

import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

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
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = tokenize_tweet(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems


vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features=1000
)

#

corpus_data_features = vectorizer.fit_transform(
    df.text.tolist() + test_df.text.tolist())

corpus_data_features_nd = corpus_data_features.toarray()

X_train = corpus_data_features_nd[0:len(df)]
Y_train = df.sentiment

X_test = corpus_data_features_nd[len(df):]
Y_test = test_df.sentiment

algo = LinearDiscriminantAnalysis()
algo.fit(X_train, Y_train)
hypotheses = algo.predict (X_test)

print "Misclassification rate is ", (1 -algo.score(X_test, Y_test))
