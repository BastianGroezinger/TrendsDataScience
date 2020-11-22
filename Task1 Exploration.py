import pandas as pd

red1 = pd.read_csv('winequality-red-1.csv', sep=';')
red2 = pd.read_csv('winequality-red-2.csv', sep=';')

white1 = pd.read_csv('winequality-white-1.csv', sep=';')
white2 = pd.read_csv('winequality-white-2.csv', sep=';')

red1 = red1.drop(['ID'], axis=1)
red2 = red2.drop(['ID'], axis=1)
redall = pd.concat([red1, red2], axis=1, sort=False)
redall = redall.drop(['flavanoids', 'magnesium', 'lightness'], axis=1)

white1 = white1.drop(['ID'], axis=1)
white2 = white2.drop(['ID'], axis=1)
whiteall = pd.concat([white1, white2], axis=1, sort=False)
whiteall = whiteall.drop(['flavanoids', 'magnesium', 'lightness'], axis=1)

wineall = pd.concat([redall, whiteall], sort=False)

print(whiteall)
print(redall)
print(wineall)
print(wineall.dtypes)
