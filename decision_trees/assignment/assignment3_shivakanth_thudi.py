import pandas as pd
import sklearn as sk
import numpy as np
import sys
from sklearn.ensemble import ExtraTreesRegressor

# Preprocessing

target_df = pd.read_csv('life expectancy by country and year.csv')
countries = target_df['Country Name'].values
target_df.loc[target_df['Country Name'] == 'Korea, Dem. People\xe2\x80\x99s Rep.', 'Country Name'] = "Korea, Dem. People's Rep."


# Impute missing Life Expectancy values in the target data using linear interpolation
target_copy = target_df.iloc[:, 1:].copy()
target_copy.columns = range(1961, 2011)
target_copy.interpolate(axis=1, limit_direction='both', method = 'linear', limit = 10000, inplace = True)
target_df.iloc[:, 1:] = target_copy.values


# Add testing data ranges
Target_Columns = ['Country Name']+ [str(year) for  year in range(1950,2017)]
target_df_final = pd.DataFrame()
for year in Target_Columns:
    if year in target_df.columns:
        target_df_final['%s' % (year,)] = target_df['%s' % (year,)]
    else:
        target_df_final['%s' % (year,)] = np.nan


# Reshape data
target_data_long = pd.melt(target_df_final, id_vars=['Country Name'], var_name = 'Year', value_name='LifeExp')


# Get the features from the file I made. The features I added were - Birth Rates, Mortality Rates, GNI, HIV rates, and Internet users.
features_final = pd.read_csv('feature_set.csv')

features_final[['Year']] = features_final[['Year']].apply(pd.to_numeric)
target_data_long[['Year']] = target_data_long[['Year']].apply(pd.to_numeric)

combined_features_target = pd.merge(features_final,target_data_long, on = ['Country Name', 'Year'])
combined_features_target = pd.get_dummies(combined_features_target)

train_range = combined_features_target.loc[(combined_features_target['Year'] > 1960) & (combined_features_target['Year'] < 2011)]


X = train_range.drop(['LifeExp'], axis = 1)
Y = train_range[['LifeExp']]

# Train the model. We picked an ExtraTrees regressor but also considered Linear Regression, Decision Trees,
# Random Forests, and Gradient Boosting machines.

regressor = ExtraTreesRegressor(n_estimators=40)
regressor.fit(X, Y.values.ravel())

# Check Training MSE if required
# predicted_y = regressor.predict(X)
# MSE = sk.metrics.mean_squared_error(Y, predicted_y)
# print "Training MSE is", MSE

# Build the feature set from the input file

input_file = sys.argv[1]
output_file = sys.argv[2]

input = pd.read_csv(input_file, header=None)
input.columns = ['Country Name', 'Year', 'GDP']

test_df = pd.concat([combined_features_target.loc[(combined_features_target['Country Name_%s' % (row['Country Name'])]==1) & (combined_features_target['Year']==row['Year'])] for index, row in input.iterrows()])

X_test_df = test_df.drop(['LifeExp'], axis = 1)

# Make the predictions
predicted_y = regressor.predict(X_test_df)

# Output them
with open(output_file, 'w') as file_handler:
    for each in predicted_y:
        file_handler.write("{}\n".format(each))