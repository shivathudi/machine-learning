import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error


date_format = "%m/%d/%y"
ref = datetime.strptime('02/01/87', date_format)

df = pd.read_csv('djia.csv')
X = pd.to_datetime(df['Date'])
Y = df.ix[:,-1]

X_days = (X - ref)
X_final = X_days.astype(pd.Timedelta).apply(lambda l: l.days)

data = pd.DataFrame()
data['X'] = X_final
data['Y'] = Y

d = range(15)
MSE_list =[]

for i in d:
    poly_params = np.polyfit(X_final, Y, i)
    MSE = mean_squared_error(Y, np.polyval(poly_params,X_final))
    MSE_list.append(MSE)

min = np.min(MSE_list)
degree = MSE_list.index(min)

plt.plot(X_final, Y, color='black')
poly_params = np.polyfit(X_final, Y, degree)
plt.plot(X_final, np.polyval(poly_params, X_final), 'r-')
plt.show()
