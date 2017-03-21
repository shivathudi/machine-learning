from tweet_processing_original import tokenize_tweet
import pandas as pd
import random
import sys
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# Reading and Processing Data
train_file = sys.argv[1]
test_file = sys.argv[2]

df_columns = ['sentiment', 'text']

df = pd.read_csv(train_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns)
df.loc[:,'text'] = df.loc[:,'text'].map(tokenize_tweet)

test_df = pd.read_csv(test_file, delimiter=',', quotechar='"', skipinitialspace=True, header=None, names=df_columns)
test_df.loc[:,'text'] = test_df.loc[:,'text'].map(tokenize_tweet)

# Prepare data for Doc2Vec format, get features from both the train and test set
# (UNSUPERVISED, USES ONLY TWEET TEXT TO GET FEATURES)

train_size = df.size / 2
test_size = test_df.size / 2

documents_train = [TaggedDocument(list(df.loc[i,'text']),[i]) for i in range(0,train_size)]
documents_test = [TaggedDocument(list(test_df.loc[i,'text']),[i+train_size]) for i in range(0,test_size)]
documents_all = documents_train+documents_test

Doc2VecTrainID = range(0,train_size+test_size)
random.shuffle(Doc2VecTrainID)
trainDoc = [documents_all[id] for id in Doc2VecTrainID]
Labels = df.loc[:,'sentiment']

# Construct the Doc2Vec model

size = 400

cores = multiprocessing.cpu_count()
model_DM = Doc2Vec(size=size, window=8, min_count=1, sample=1e-4, negative=5, workers=cores,  dm=1, dm_concat=1 )
model_DBOW = Doc2Vec(size=size, window=8, min_count=1, sample=1e-4, negative=5, workers=cores, dm=0)

model_DM.build_vocab(trainDoc)
model_DBOW.build_vocab(trainDoc)


# We pass through the data set multiple times,
# shuffling the training data each time to improve accuracy.

for epoch in range(0,10):
    random.shuffle(Doc2VecTrainID)
    trainDoc = [documents_all[id] for id in Doc2VecTrainID]
    model_DM.train(trainDoc)
    model_DBOW.train(trainDoc)


#Fit the model and make predictions

train_targets, train_regressors = zip(*[(Labels[id], list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id])) for id in range(0,train_size)])
train_regressors = sm.add_constant(train_regressors)
predictor = LogisticRegression(multi_class='multinomial',solver='lbfgs')
predictor.fit(train_regressors,train_targets)


test_regressors = [list(model_DM.docvecs[id])+list(model_DBOW.docvecs[id]) for id in range(0,test_size)]
test_regressors = sm.add_constant(test_regressors)
test_predictions = predictor.predict(test_regressors)


y_test = test_df.loc[:,'sentiment']
print 'Test Accuracy: %.2f'%predictor.score(test_regressors, y_test)