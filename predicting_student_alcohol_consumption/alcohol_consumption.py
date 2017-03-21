
# coding: utf-8

# In[27]:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import sklearn

from StringIO import StringIO
import pydotplus




# In[3]:

def dummy_variable(columns, dataframe):
    enc = OneHotEncoder()
    labler = LabelEncoder()
    
    for colname in columns:
        labels = labler.fit_transform(dataframe[colname])
        labels = labels.reshape(-1,1)
        categor_vars = enc.fit_transform(labels)
        categor_vars = categor_vars.toarray()
        df = pd.DataFrame(categor_vars)
        columns = df.columns
        column_labels = labler.inverse_transform(columns)
        df.columns = [str(col) + "_" + colname for col in column_labels]

        del dataframe[colname]
        dataframe = pd.concat([dataframe.reset_index(drop=True), df.iloc[:,:-1].reset_index(drop=True)], axis=1)
    
    return dataframe 

def cross_val(estimator, x, y, k=10, reg=True):
    
    
    
    kf = KFold(n_splits=k, shuffle=True)
    score = []
    for train_index, test_index in kf.split(X):
        X_train, Y_train = x.iloc[train_index,:], y[train_index] 
        X_test, Y_test = x.iloc[test_index,:], y[test_index]
        
        estimator.fit(X_train, Y_train)
        y_predict = estimator.predict(X_test)
        if reg:
            score.append(mean_squared_error(Y_test, y_predict))
        else:
            score.append(accuracy_score(Y_test, y_predict))
    return np.mean(score)


# In[4]:

port = pd.read_csv('student-por.csv', sep=";")
port['class'] = "portuguese"
math = pd.read_csv('student-mat.csv', sep=";")
math['class'] = 'math'
data = pd.concat([port, math], axis=0)
data = data.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
data = data.reset_index(drop=True)


# In[56]:

Y = data['Walc']

X = data.drop(['Walc'], axis=1)
columns = [u'school', u'sex', u'famsize', u'Pstatus', u'Medu',
       u'Fedu', u'Mjob', u'Fjob', u'reason', u'guardian', u'traveltime',
       u'studytime', u'failures', u'schoolsup', u'famsup', u'paid',
       u'activities', u'nursery', u'higher', u'internet', u'romantic',
       u'famrel', u'freetime', u'goout','health', 'class','Dalc', 'G2']
X = data.loc[:,columns]
X = dummy_variable(columns, X)

extra_col = ['failures', 'absences', 'age']
X = pd.concat([X, data.loc[:,extra_col]], axis=1)

print X.columns[78]


# In[91]:
tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
tree.fit(X, Y)
feature_importances = pd.Series(tree.feature_importances_)

my_tree = tree
dotfile = StringIO()
sklearn.tree.export_graphviz(my_tree, out_file=dotfile)
graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("tree.pdf")

#print X.columns[feature_importances[feature_importances >0].index] 

X = X.iloc[:,feature_importances[feature_importances >0].index]
# In[]
k = 5
print "KNN Accuracy =\t\t\t", cross_val(estimator=KNeighborsClassifier(n_neighbors=k), x=X, y=Y, reg=False)
# In[57]:

print "LDA Accuracy =\t\t\t", cross_val(estimator=LinearDiscriminantAnalysis(), x=X, y=Y, reg=False)


# In[67]:

print "Decision Tree Accuracy =\t", cross_val(estimator=DecisionTreeClassifier(criterion='gini', max_depth=4), x=X, y=Y, reg=False)


# In[74]:

from sklearn.ensemble import RandomForestClassifier
estimators=25
print "Random Forest Accuracy =\t", cross_val(estimator=RandomForestClassifier(n_estimators = estimators, random_state=False ,class_weight='auto'), x=X, y=Y, reg=False)


# In[85]:

rate = .7
ada = AdaBoostClassifier(learning_rate=rate, random_state=False)
print "ADA Boost Accuracy =\t\t", cross_val(estimator=ada, x=X, y=Y, reg=False)


# In[61]:


mlp = MLPClassifier(solver='lbfgs', random_state=1, hidden_layer_sizes=(40,40, 40))
print "MLP ANN Accuracy =\t\t", cross_val(estimator=mlp, x=X, y=Y, reg=False)





