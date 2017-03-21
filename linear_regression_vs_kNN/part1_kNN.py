import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor

k_list = range(1,457)
k_MSE_tuples = []

df = pd.read_csv('boston.csv')
cols = df.columns.tolist()

X = df[cols[:-1]]
y = df[cols[-1]]

X_train = X.iloc[:(len(df)-50),]
y_train = y.iloc[:(len(df)-50),]

X_test = X.iloc[(len(df)-50):,]
y_test = y.iloc[(len(df)-50):,]

for k in k_list:
    algo = KNeighborsRegressor(n_neighbors=k)
    algo.fit(X_train, y_train)
    predictions = algo.predict(X_test)
    MSE = sk.metrics.mean_squared_error(y_test, predictions)
    k_MSE_tuples.append((k,MSE))


lowest_MSE = None

for each_tuple in k_MSE_tuples:
    if lowest_MSE is None:
        best_k = each_tuple[0]
        lowest_MSE = each_tuple[1]
    else:
        if each_tuple[1] < lowest_MSE:
            best_k = each_tuple[0]
            lowest_MSE = each_tuple[1]

#Print the best MSE and k

print "Lowest MSE for Boston dataset using k-nearest neighbors is %s, with k as %s." \
      "\n(Considering k-values in the range of 1 to %s)" % (lowest_MSE, best_k, k_list[-1])

# print "MSE for Boston Dataset using Linear Regression is" , lowest_MSE