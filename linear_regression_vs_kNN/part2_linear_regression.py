import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn.preprocessing import Imputer
import sys
import string

# Reading in the data
print "loading data..."

# EXAMPLE USAGE : "USW00023234" "USW00014918" "USW00012919" "USW00013743" "USW00025309"
test_weather_stations = sys.argv[1:len(sys.argv)]
hardcoded_test_stations_to_be_excluded = ["USW00023234","USW00014918","USW00012919","USW00013743","USW00025309"]

df_columns = ['ID', 'Month', 'Day'] + ['Hour0%s' % (i) for i in range(0,10)] + ['Hour%s' % (i) for i in range(10,24)]
df = pd.read_table('hly-temp-normal.txt', header=None, names=df_columns, delim_whitespace=True)

# Cleaning
print "cleaning and splitting data..."
all=string.maketrans('','')
nodigs=all.translate(all, string.digits)

def clean(record):
    if record == "-9999":
        result = np.nan
    else:
        result = int(record.translate(all, nodigs))

    return result


for col in df_columns[3:]:
    df[col] = df[col].map(lambda x: clean(x))

# Splitting the data into train and test sets
test_data = df.ix[df['ID'].isin(test_weather_stations)].copy()
train_data = df.ix[~df['ID'].isin(hardcoded_test_stations_to_be_excluded)].copy()

# Free some memory
del df

# Imputations
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_data.loc[:,df_columns[3:]] = imp.fit_transform(train_data.loc[:,df_columns[3:]])
test_data.loc[:,df_columns[3:]] = imp.fit_transform(test_data.loc[:,df_columns[3:]])

# Reshaping from wide to long format
print "reshaping data..."
test_data_long = pd.melt(test_data, id_vars=['ID', 'Month', 'Day'], var_name = 'Hour', value_name='Temp')
train_data_long = pd.melt(train_data, id_vars=['ID', 'Month', 'Day'], var_name = 'Hour', value_name='Temp')

# Free some more memory
del train_data
del test_data

# Sort values
train_data_long.sort_values(['ID', 'Month', 'Day', 'Hour'], inplace=True)
test_data_long.sort_values(['ID', 'Month', 'Day', 'Hour'], inplace=True)

train_data_long.reset_index(drop=True,inplace=True)
test_data_long.reset_index(drop=True,inplace=True)

# Perform cyclic shift and get first feature, PreviousHourTemp
print "creating first feature..."

new_col = train_data_long.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True)
train_data_long['PreviousHourTemp'] = new_col

new_col = test_data_long.groupby(['ID']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True)
test_data_long['PreviousHourTemp'] = new_col

train_data_long.reset_index(drop=True,inplace=True)
test_data_long.reset_index(drop=True,inplace=True)

# Sort values again, differently
train_data_long.sort_values(['ID', 'Hour', 'Month', 'Day'], inplace=True)
test_data_long.sort_values(['ID', 'Hour', 'Month', 'Day'], inplace=True)

train_data_long.reset_index(drop=True,inplace=True)
test_data_long.reset_index(drop=True,inplace=True)

# Perform cyclic shift and get second feature, PreviousDayTemp
print "creating second feature..."

new_col = train_data_long.groupby(['ID', 'Hour']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True)
train_data_long['PreviousDayTemp'] = new_col

new_col = test_data_long.groupby(['ID', 'Hour']).apply(lambda x: x.reindex(index = np.roll(x.index, 1)))['Temp'].reset_index(drop=True)
test_data_long['PreviousDayTemp'] = new_col

train_data_long.reset_index(drop=True,inplace=True)
test_data_long.reset_index(drop=True,inplace=True)

# Get the third feature, MeanTempAll
print "creating third feature..."

train_data_long['MeanTempAll'] = train_data_long.groupby(['Month', 'Day', 'Hour'])['Temp'].transform('mean')
test_data_long['MeanTempAll'] = test_data_long.groupby(['Month', 'Day', 'Hour'])['Temp'].transform('mean')

# Get the fourth feature, MeanTempPer
print "creating fourth and final feature...(this will take some time due to shift - calculating mean over previous rows)"

f = lambda x: x.expanding(min_periods=1).mean().shift(1)
g = lambda x: x.fillna(x.mean())

train_data_long['MeanTempPer'] = train_data_long.groupby(['ID','Month', 'Day'])['Temp'].transform(f)
train_data_long['MeanTempPer'] = train_data_long.groupby(['ID','Month', 'Day'])['MeanTempPer'].transform(g)

test_data_long['MeanTempPer'] = test_data_long.groupby(['ID', 'Month', 'Day'])['Temp'].transform(f)
test_data_long['MeanTempPer'] = test_data_long.groupby(['ID', 'Month', 'Day'])['MeanTempPer'].transform(g)

# Begin fitting and testing model
print "fitting the model..."

algo = linear_model.LinearRegression()

X_train = train_data_long[['PreviousHourTemp', 'PreviousDayTemp', 'MeanTempAll', 'MeanTempPer']]
y_train = train_data_long[['Temp']]

X_test = test_data_long[['PreviousHourTemp', 'PreviousDayTemp', 'MeanTempAll', 'MeanTempPer']]
y_test = test_data_long[['Temp']]

algo.fit (X_train, y_train)

print "making predictions..."
predictions = algo.predict(X_test)

MSE = sk.metrics.mean_squared_error(y_test, predictions)

print "------------------------------------------------------"
print "MSE for Climate data using linear regression is" , MSE