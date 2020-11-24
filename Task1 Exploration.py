import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn import preprocessing

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

#Initialize MaxAbsScaler()
scaler = preprocessing.MaxAbsScaler()
#fit wine np.array to MaxAbsScaler
scaler.fit(wine)
#Transform wine np.array to scaled data
wine = scaler.transform(wine)

#transform np.array to Dataframe
wineall = pd.DataFrame({'ID': wine[:,0], 'fixed acidity': wine[:,1], 'volatile acidity': wine[:,2], 'citric acid': wine[:,3], 'residual sugar': wine[:,4], 'chlorides': wine[:,5], 'flavanoids': wine[:,6], 'free sulfur dioxide': wine[:,7],'total sulfur dioxide': wine[:,8], 'density': wine[:,9], 'pH': wine[:,10], 'sulphates': wine[:,11],'magnesium': wine[:,12], 'alcohol': wine[:,13], 'lightness': wine[:,14], 'quality': wine[:,15]})

# Output of different data
print("Alle Weine\n", wineall)
print("Feature Datentypen\n", wineall.dtypes)

wineall = wineall.drop(['ID'], axis=1)
wineall.to_csv('data/wineall.csv')