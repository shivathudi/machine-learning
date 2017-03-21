import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline


date_format = "%m/%d/%y"
ref = datetime.strptime('02/01/87', date_format)

df = pd.read_csv('djia.csv')
X = pd.to_datetime(df['Date'])
Y = df.ix[:,-1]

X_days = (X - ref)
X = X_days.astype(pd.Timedelta).apply(lambda l: l.days)

data = pd.DataFrame()
data['X'] = X
data['Y'] = Y

smoothing = 5
knots = len(Y)

s = UnivariateSpline(X,Y, k=smoothing, s=knots)

print s.get_residual()



