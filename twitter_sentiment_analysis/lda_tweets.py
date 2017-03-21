import csv
import sys
import nltk
# import nltk.tokenize.casual
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)

train_file = sys.argv[1]
test_file = sys.argv[2]

with open (train_file, "rb") as my_file:
    reader = csv.reader(my_file, delimiter=',', quotechar='"', skipinitialspace=True)
    tweets = []
    for row in reader:
        tweets.append([row[0], row[1]])

with open (test_file, "rb") as my_file:
    reader = csv.reader(my_file, delimiter=',', quotechar='"', skipinitialspace=True)
    test_tweets = []
    for row in reader:
        test_tweets.append([row[0], row[1]])


regex_str = [
    r'(?:[+\-]?\d+[,/.:-]\d+[+\-]?)',  # Numbers, including fractions, decimals.
    r"(?:[a-z][a-z'\-_]+[a-z])", # Words with apostrophes or dashes.
    r'(?:[\w_]+)',  # Words without apostrophes or dashes.
    r'(?:\S)'  # Everything else that isn't whitespace.
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return [token.lower() for token in tokens_re.findall(s)]

function_words = ['i', 'the', 'and', 'to', 'a', 'of', 'that', 'in',
                  'it', 'my', 'is', 'you', 'was', 'for', 'have', 'with', 'he', 'me',
                  'on', 'but']

punct_words = ['.', ',', '!']

Y_train = [tweet[0] for tweet in tweets]
Y_test = [tweet[0] for tweet in test_tweets]

X_train =[]
X_test = []

for tweet in tweets:
    text = tweet[1]
    text.replace('.', '. ')

    try:
        tokenized_list = word_tokenize(text)
    except:
        tokenized_list =tokenize(text)

    counts = Counter(tokenized_list)

    row_vector = []
    for word in function_words + punct_words:
        numerator = 0.0
        denominator = float(len(tokenized_list))
        numerator += counts[word]
        row_vector.append(numerator/denominator)

    X_train.append(row_vector)
    # X_train.append([counts[word]/float(len(tokenized_list)) for word in function_words + punct_words])

for tweet in test_tweets:
    text = tweet[1]
    text.replace('.', '. ')

    try:
        tokenized_list = word_tokenize(text)
    except:
        tokenized_list = tokenize(text)

    counts = Counter(tokenized_list)

    row_vector = []
    for word in function_words + punct_words:
        numerator = 0.0
        denominator = float(len(tokenized_list))
        numerator += counts[word]
        row_vector.append(numerator/denominator)

    X_test.append(row_vector)
    # X_train.append([counts[word]/float(len(tokenized_list)) for word in function_words + punct_words])


algo = LinearDiscriminantAnalysis()
algo.fit(X_train, Y_train)
hypotheses = algo.predict (X_test)

print "Misclassification rate is ", (1 -algo.score(X_test, Y_test))

def get_words_in_tweets(tweets):
    all_words = []
    for tweet in tweets:
      all_words.extend(tweet[1])
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

# word_features = get_word_features(get_words_in_tweets(tweets))
#
# print get_words_in_tweets(tweets)
