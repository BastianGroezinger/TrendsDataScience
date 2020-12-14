import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest

# reading in redwine data
red1 = pd.read_csv('data/winequality-red-1.csv', sep=';', decimal =',')
red2 = pd.read_csv('data/winequality-red-2.csv', sep=';', decimal =',')

#reading in whitewine date
white1 = pd.read_csv('data/winequality-white-1.csv', sep=';', decimal =',')
white2 = pd.read_csv('data/winequality-white-2.csv', sep=';', decimal =',')

#droping column id from redwine.csv2
red2 = red2.drop(['ID'], axis=1)
#merging redwine data
redall = pd.concat([red1, red2], axis=1, sort=False)
#droping last three rows because they were empty
redall = redall.drop([1598, 1599, 1600])

#droping column id from whitewine.csv2
white2 = white2.drop(['ID'], axis=1)
#merging whitewine data
whiteall = pd.concat([white1, white2], axis=1, sort=False)

#merging redwine and whitewine data
wineall = pd.concat([redall, whiteall], sort=False)

#Initialize Deterministic Regression Imputation
imp = IterativeImputer(max_iter = 10, sample_posterior = False)
#create new np.array without missing values
wine = np.round(imp.fit_transform(wineall,1),2)

X, y = wine[:, :-1], wine[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# evaluate on raw dataset
print("Shape of dataset: ",X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

# identify outliers in the training dataset Isolation forrest
iso = IsolationForest(contamination=0.1)
y_out = iso.fit_predict(X_train)
# select all rows that are not outliers (inlier=1, outlier=-1)
mask = y_out != -1
X_train_red, y_train_red = X_train[mask, :], y_train[mask]
# Inliers and Outliers output
print("Inliers: ",X_train_red.shape[0],"Outliers",X_train.shape[0]-X_train_red.shape[0])
# Model fitting
model = LinearRegression()
model.fit(X_train_red, y_train_red)
# evaluation of model
y_pred = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

# Inliers and Outliers output Local Outlier Factor
print("Inliers: ",X_train_red.shape[0],"Outliers",X_train.shape[0]-X_train_red.shape[0])
# fitting the model
model = LinearRegression()
model.fit(X_train_red, y_train_red)
# evaluate of model
y_pred = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_pred, yhat)
print('MAE: %.3f' % mae)

# Inliers and Outliers output One-Class SVM
print("Inliers: ",X_train_red.shape[0],"Outliers",X_train.shape[0]-X_train_red.shape[0])
# fitting the model
model = LinearRegression()
model.fit(X_train_red, y_train_red)
# evaluate the model
y_pred = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

