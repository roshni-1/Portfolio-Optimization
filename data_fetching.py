import yfinance as yf

# Define the list of stocks you want to analyze
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META','NVDA','ASML','TSLA','APLT','NNOX','RYZB','COIN','ALCE','HTOO','LIVE','SOFI','DBX','BLZE','INTR','GXAI','MGX','IRDM','ZCARW','NDAQ','SMCI']

# Set the start and end dates for data retrieval
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch historical stock prices from Yahoo Finance
stock_data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = stock_data.pct_change()

# Display the first few rows of the returns dataframe
print(returns.head())
