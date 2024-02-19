#Importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import yfinance as yf

stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META','NVDA','ASML','TSLA','APLT','NNOX','RYZB','COIN','ALCE','HTOO','LIVE','SOFI','DBX','BLZE','INTR','GXAI','MGX','IRDM','ZCARW','NDAQ','SMCI']

start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

stock_data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

returns = stock_data.pct_change()

print(returns.head())
