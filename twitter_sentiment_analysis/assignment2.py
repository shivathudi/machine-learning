import csv
import sys
import nltk
# import nltk.tokenize.casual
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

train_file = sys.argv[1]
test_file = sys.argv[2]

with open (train_file, "rb") as my_file:
    reader = csv.reader(my_file, delimiter=',', quotechar='"', skipinitialspace=True)
    tweets = []
    for row in reader:
        tweets.append([row[0], row[1]])


regex_str = [
    r'(?:[+\-]?\d+[,/.:-]\d+[+\-]?)',  # Numbers, including fractions, decimals.
    r"(?:[a-z][a-z'\-_]+[a-z])", # Words with apostrophes or dashes.
    r'(?:[\w_]+)',  # Words without apostrophes or dashes.
    r'(?:\S)'  # Everything else that isn't whitespace.
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return [token.lower() for token in tokens_re.findall(s)]

function_words = [('i', "i'm"), 'the', 'and', 'to', 'a', 'of', 'that', 'in',
                  'it', 'my', 'is', 'you', 'was', 'for', 'have', 'with', 'he', 'me',
                  'on', 'but']

punct_words = ['.', ',', '!']

Y_train = [tweet[0] for tweet in tweets]
X_train =[]

for tweet in tweets:
    text = tweet[1]
    tokenized_list = tokenize(text)
    counts = Counter(tokenized_list)

    row_vector = []
    for tuple in function_words + punct_words:
        numerator = 0.0
        denominator = float(len(tokenized_list))
        for word in tuple:
            numerator += counts[word]
        row_vector.append(numerator/denominator)

    X_train.append(row_vector)

    # X_train.append([counts[word]/float(len(tokenized_list)) for word in function_words + punct_words])



