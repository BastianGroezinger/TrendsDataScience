import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

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

x = wine[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
y = wine[:,15]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print("Training data: ",x_train.shape, y_train.shape)
print("Test data: ",x_test.shape, y_test.shape)

#Fit SVM classifier to training data
clf = svm.SVR(kernel='linear', C=1).fit(x_train, y_train)
#Calculate accuracies
print("Training data accuracy: ",round(clf.score(x_train, y_train),3))
print("Test data accuracy: ",round(clf.score(x_test, y_test),3))

