import yfinance as yf
import pandas as pd

TICKER = 'MSFT'
START = '2017-01-01'
END = '2025-07-01'

stock = yf.Ticker(TICKER)
df = stock.history(start=START, end=END)

df = df[['Close']]
df.dropna(inplace=True)

df.to_csv('stock_data.csv')
print('Data saved to stock_data.csv')
