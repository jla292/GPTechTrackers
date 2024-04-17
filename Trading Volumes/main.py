import pandas as pd
import matplotlib.pyplot as plt

nvda = pd.read_csv('Trading Volumes/NVDA.csv')
print(nvda.head())

x = nvda['Date']
y = nvda['Close']

plt.plot(x, y)

# x = nvda['Date']
# y = nvda['Volume']

# plt.figure()
# plt.plot(x, y)

msft = pd.read_csv('Trading Volumes/MSFT.csv')
msft.head()

meta = pd.read_csv('Trading Volumes/META.csv')
meta.head()