from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import GridSearchCV   #Performing grid search
from adaboost import *


train_file = 'hw2_data/spambase.train'
test_file = 'hw2_data/spambase.test'

X, Y = parse_spambase_data(train_file)
X_test, Y_test = parse_spambase_data(test_file)

grd = GradientBoostingClassifier(learning_rate=0.1, max_depth=6, n_estimators=400)

grd.fit(X,Y)

print grd.score(X,Y)

print grd.score(X_test, Y_test)


