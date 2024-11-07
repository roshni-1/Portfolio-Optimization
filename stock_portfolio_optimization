import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Function to fetch historical stock data
def get_stock_data(stock_symbols):
    stock_data = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # Fetch 1 year of data
        if not data.empty:
            stock_data[symbol] = data
        else:
            print(f"No data available for {symbol}")
    return stock_data

# Function for risk analysis
def calculate_risk_metrics(stock_data):
    risk_metrics = {}
    for symbol, data in stock_data.items():
        if 'Close' not in data.columns:
            print(f"Error: 'Close' price not available for {symbol}")
            continue
        
        data['Daily Return'] = data['Close'].pct_change()
        volatility = data['Daily Return'].std() * np.sqrt(252)  # Annualized volatility
        running_max = data['Close'].cummax()
        drawdown = (data['Close'] - running_max) / running_max
        max_drawdown = drawdown.min()
        sharpe_ratio = data['Daily Return'].mean() / data['Daily Return'].std() * np.sqrt(252)

        risk_metrics[symbol] = {
            'Volatility': volatility,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }
    return risk_metrics

# Function to calculate portfolio performance
def calculate_portfolio_performance(stock_data, quantities, purchase_prices):
    portfolio_values = pd.DataFrame()
    for symbol, data in stock_data.items():
        if 'Close' not in data.columns:
            continue

        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod() - 1
        data['Investment Value'] = quantities[symbol] * data['Close']
        portfolio_values[symbol] = data['Investment Value']

        plt.plot(data.index, data['Cumulative Return'], label=f"Cumulative Return: {symbol}")

    if not portfolio_values.empty:
        portfolio_values['Total Portfolio Value'] = portfolio_values.sum(axis=1)
        plt.plot(portfolio_values.index, portfolio_values['Total Portfolio Value'], label="Total Portfolio Value", color="black", linewidth=2)

    plt.title("Portfolio Performance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value (INR)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return portfolio_values

# Function to create candlestick chart for each stock
def plot_candlestick(stock_data):
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates

    for symbol, data in stock_data.items():
        data = data[['Open', 'High', 'Low', 'Close']].dropna()
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].map(mdates.date2num)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        candlestick_ohlc(ax, data[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.6, colorup='green', colordown='red')
        ax.xaxis_date()
        ax.set_title(f"{symbol} Candlestick Chart")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.grid(True)
        plt.show()

# Portfolio Optimization Recommendation
def portfolio_recommendation(risk_metrics, quantities):
    recommendations = {}
    for symbol, metrics in risk_metrics.items():
        if metrics['Sharpe Ratio'] > 1:
            recommendation = f"Buy more stocks (ideal to buy up to {quantities[symbol] * 1.2:.0f} shares)."
        elif metrics['Sharpe Ratio'] < 0.5 and metrics['Max Drawdown'] < -0.2:
            recommendation = f"Consider selling {int(quantities[symbol] * 0.5)} shares due to high risk."
        else:
            recommendation = "Hold the stock for now."
        
        recommendations[symbol] = recommendation
    return recommendations

# Get stock data for user input
input_symbols = input("Enter stock symbols with suffix (e.g., TATASTEEL.NS, RPOWER.BS): ")
stock_symbols = [symbol.strip() for symbol in input_symbols.split(",")]

input_quantities = input("Enter quantities for each stock symbol (e.g., 2, 2, 5): ")
quantities_list = [int(q.strip()) for q in input_quantities.split(",")]
if len(stock_symbols) != len(quantities_list):
    print("Error: Number of quantities does not match the number of stock symbols.")
else:
    quantities = dict(zip(stock_symbols, quantities_list))

purchase_prices = {}
for symbol in stock_symbols:
    purchase_prices[symbol] = float(input(f"Enter purchase price for {symbol}: "))

# Fetch stock data
stock_data = get_stock_data(stock_symbols)

# Plot candlestick charts
plot_candlestick(stock_data)

# Calculate risk metrics
risk_metrics = calculate_risk_metrics(stock_data)

# Calculate and plot portfolio performance
portfolio_values = calculate_portfolio_performance(stock_data, quantities, purchase_prices)

# Portfolio Optimization Recommendations
recommendations = portfolio_recommendation(risk_metrics, quantities)
for symbol, recommendation in recommendations.items():
    print(f"\nRecommendation for {symbol}: {recommendation}")
