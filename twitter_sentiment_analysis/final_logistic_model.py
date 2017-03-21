import pandas as pd
import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer


# Reading and Processing Data
train_file = sys.argv[1]
test_file = sys.argv[2]

df_columns = ['sentiment', 'text']

df = pd.read_csv(train_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns,encoding='ISO-8859-1')
test_df = pd.read_csv(test_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns,encoding='ISO-8859-1')


stemmer = SnowballStemmer("english")

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.tokenize.casual.TweetTokenizer().tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

# Since we set stop_words = None, function word appearances such as 'I', 'me', 'you' will be included in the features

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = None,
    ngram_range= (1,3),
    max_features=125000
)

# Build the corpus

corpus_data_features = vectorizer.fit_transform(
    df.text.tolist() + test_df.text.tolist())

corpus_data_features_nd = corpus_data_features.toarray()


# Fit the model and check performance on testing set

X_train = corpus_data_features_nd[0:len(df)]
Y_train = df.sentiment

X_test = corpus_data_features_nd[len(df):]
Y_test = test_df.sentiment

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=Y_train)

y_pred = log_model.predict(X_test)

print "---------------------------------------------------------------"
print "MISCLASSIFICATION RATE = ", (1 -log_model.score(X_test, Y_test))

print "---------------------------------------------------------------"
print(classification_report(Y_test, y_pred))