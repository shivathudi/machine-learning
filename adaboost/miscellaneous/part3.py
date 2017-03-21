from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import GridSearchCV   #Performing grid search
from adaboost import *


train_file = 'hw2_data/spambase.train'
test_file = 'hw2_data/spambase.test'

X, Y = parse_spambase_data(train_file)

param_test1 = {'n_estimators':[400], 'max_depth':[6,7], 'learning_rate' : [ 0.1, 0.25]}

gsearch = GridSearchCV(estimator = GradientBoostingClassifier(),
                       param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)

gsearch.fit(X,Y)

results = gsearch.cv_results_

print results

print gsearch.best_params_, gsearch.best_score_

