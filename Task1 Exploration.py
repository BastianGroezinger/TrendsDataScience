import pandas as pd
import os

red1 = pd.read_csv('data/winequality-red-1.csv', sep=';', decimal =',')
red2 = pd.read_csv('data/winequality-red-2.csv', sep=';', decimal =',')

white1 = pd.read_csv('data/winequality-white-1.csv', sep=';', decimal =',')
white2 = pd.read_csv('data/winequality-white-2.csv', sep=';', decimal =',')

red2 = red2.drop(['ID'], axis=1)
redall = pd.concat([red1, red2], axis=1, sort=False)
redall = redall.drop([1598, 1599, 1600])

white2 = white2.drop(['ID'], axis=1)
whiteall = pd.concat([white1, white2], axis=1, sort=False)

wineall = pd.concat([redall, whiteall], sort=False)

print("Weißwein\n", whiteall)
print("Rotwein\n", redall)
print("Alle Weine\n", wineall)
print("Feature Datentypen\n", wineall.dtypes)

wineall = wineall.drop(['ID'], axis=1)
wineall.to_csv('data/wineall.csv')