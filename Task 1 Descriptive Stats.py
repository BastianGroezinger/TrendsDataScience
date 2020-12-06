import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

pd.options.display.width = 0
pd.options.display.float_format = '{:.4f}'.format

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
redall['winetype'] = 'redwine'
redall = redall.drop(['ID'], axis=1)
#droping last three rows because they were empty
#redall = redall.drop([1598, 1599, 1600])

#merging whitewine data
whiteall = pd.concat([white1, white2], axis=1, sort=False)
whiteall['winetype'] = 'whitewine'
whiteall = whiteall.drop(['ID'], axis=1)

#merging redwine and whitewine data
wineall = pd.concat([redall, whiteall], sort=False)

print('---------------------------------------------------------------------------------------------------')
print("Shape", wineall.shape)
print('---------------------------------------------------------------------------------------------------')

print('MV?', wineall.isnull().values.any())
print('---------------------------------------------------------------------------------------------------')
#print('MV für jedes feature\n', wineall.isnull().sum())
print('---------------------------------------------------------------------------------------------------')
print('Total count MV\n', wineall.isnull().sum().sum())
print('---------------------------------------------------------------------------------------------------')

#print('Leere Zellen\n', wineall['fixed acidity'].isnull())
print('---------------------------------------------------------------------------------------------------')

# Output of different data
print("Alle Weine\n", wineall)
print('---------------------------------------------------------------------------------------------------')
print("Feature Datentypen\n", wineall.dtypes)

wineall.to_csv('data/wineall.csv')
print('---------------------------------------------------------------------------------------------------')
#descriptive statistics raw data
#print('Whitewine data\n', whiteall.describe())
print('----------------------------------------------------------------------------------------------------')
#print('All Wines data\n', wineall.describe())
print('----------------------------------------------------------------------------------------------------')
#print('Redwine data\n', redall.describe())
print('----------------------------------------------------------------------------------------------------')

total = wineall.isnull().sum().sort_values(ascending=False)
percent = (wineall.isnull().sum()/wineall.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print('All Wine missing data:\n', missing_data)

print('----------------------------------------------------------------------------------------------------')

print('Indices of MV e.g. pH \n', wineall[wineall['pH'].isnull()].index.tolist())

#visual analytics - correlation Matrix between features
#corrMatrixAll = wineall.corr()
#corrMatrixWhite =  whiteall.corr()
corrMatrixRed = redall.corr()

#scatter_matrix(corrMatrix, figsize=(16,12), alpha=0.3)
#sn.heatmap(corrMatrixAll, annot=True, cmap="coolwarm")
#sn.heatmap(corrMatrixWhite, annot=True, cmap="coolwarm")
sn.heatmap(corrMatrixRed, annot=True, cmap="coolwarm")

#style
#corrMatrixAll.style.background_gradient(axis= None)
#corrMatrixWhite.style.background_gradient(axis= None)
corrMatrixRed.style.background_gradient(axis= None)

#plt.show()
#plt.show()
plt.show()

#scatterplot

#sn.scatterplot(y=wineall['free sulfur dioxide'], x=wineall['chlorides'],palette=['green','orange'])

#plt.show()

#Boxplot categorial white and redwine
#ax = sn.boxplot(x = 'winetype', y = 'quality', hue="winetype", data=wineall, palette="Set3")

#plt.show()


