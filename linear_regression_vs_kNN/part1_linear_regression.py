import pandas as pd
import sklearn as sk
from sklearn import linear_model


df = pd.read_csv('boston.csv')

algo = linear_model.LinearRegression()

cols = df.columns.tolist()

X = df[cols[:-1]]
y = df[cols[-1]]

X_train = X.iloc[:(len(df)-50),]
y_train = y.iloc[:(len(df)-50),]

X_test = X.iloc[(len(df)-50):,]
y_test = y.iloc[(len(df)-50):,]


algo.fit (X_train, y_train)
predictions = algo.predict(X_test)

MSE = sk.metrics.mean_squared_error(y_test, predictions)

print "MSE for Boston dataset using linear regression is" , MSE