import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from datetime import datetime
import warnings
import seaborn as sns
from textblob import TextBlob
import feedparser
import random
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to fetch stock data
def get_stock_data(symbols, period="5y"):
    stock_data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        data.index = data.index.tz_localize(None)  # Remove timezone
        stock_data[symbol] = data
    return stock_data

# Function to fetch index data (e.g., NIFTY 50 or SENSEX)
def get_index_data(index_symbol="^NSEI", period="5y"):
    index = yf.Ticker(index_symbol)
    index_data = index.history(period=period)
    index_data.index = index_data.index.tz_localize(None)
    return index_data

# Function to compare stock performance to an index
def compare_with_index(stock_data, index_data):
    comparison = {}
    for symbol, data in stock_data.items():
        stock_return = data['Close'].pct_change().mean() * 252
        index_return = index_data['Close'].pct_change().mean() * 252
        alpha = stock_return - index_return
        comparison[symbol] = {'Stock Return': stock_return, 'Index Return': index_return, 'Alpha': alpha}
    return comparison

# Portfolio Allocation (using Risk and Return)
def portfolio_allocation(stock_data, quantities, invested_amounts):
    returns = pd.DataFrame({symbol: data['Close'].pct_change() for symbol, data in stock_data.items()}).dropna()
    cov_matrix = returns.cov()
    avg_returns = returns.mean()
    weights = np.array([invested_amounts[symbol] for symbol in stock_data])
    weights /= weights.sum()
    portfolio_return = np.dot(weights, avg_returns) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk
    return portfolio_return, portfolio_risk, sharpe_ratio

# Profit/Loss Calculation
def calculate_profit_loss(stock_data, quantities, invested_amounts):
    profit_loss = {}
    for symbol, data in stock_data.items():
        current_price = data['Close'].iloc[-1]
        invested_value = invested_amounts[symbol]
        current_value = current_price * quantities[symbol]
        profit_loss[symbol] = {
            'Current Price': current_price,
            'Invested Value': invested_value,
            'Current Value': current_value,
            'Profit/Loss': current_value - invested_value,
            'Profit/Loss %': ((current_value - invested_value) / invested_value) * 100
        }
    return profit_loss

# Sentiment Analysis using Google RSS Feed
def analyze_sentiment(stock_symbol):
    rss_url = f"https://news.google.com/rss/search?q={stock_symbol}&hl=en-IN&gl=IN&ceid=IN%3Aen"
    try:
        news_data = feedparser.parse(rss_url)
        
        if len(news_data.entries) == 0:
            return "No Articles Found"
        
        sentiment_score = 0
        for article in news_data.entries:
            text = (article.title or '') + ' ' + (article.summary or '')
            sentiment_score += TextBlob(text).sentiment.polarity
        
        avg_sentiment = sentiment_score / len(news_data.entries) if len(news_data.entries) > 0 else 0
        
        if avg_sentiment > 0:
            return "Positive Sentiment"
        elif avg_sentiment < 0:
            return "Negative Sentiment"
        else:
            return "Neutral Sentiment"
    
    except Exception as e:
        print(f"Error occurred while fetching or processing news for {stock_symbol}: {e}")
        return "Error Fetching Sentiment"

# Forecast stock prices using Prophet
def forecast_stock_prices(stock_symbol):
    data = get_stock_data_for_forecast(stock_symbol)
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=5)  # Forecast for the next 5 days
    forecast = model.predict(future)
    return forecast

# Forecast stock data for a symbol (for Prophet)
def get_stock_data_for_forecast(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1y")
    data = data[['Close']].reset_index()
    data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)  # Remove timezone if present
    return data

# Monte Carlo Simulation for Stock Price Prediction
def monte_carlo_simulation(stock_data, symbol, num_simulations=1000, days=5):
    last_price = stock_data[symbol]['Close'].iloc[-1]
    returns = stock_data[symbol]['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    simulations = np.zeros(num_simulations)
    for i in range(num_simulations):
        random_returns = np.random.normal(mean_return, std_return, days)
        price_path = last_price * np.cumprod(1 + random_returns)
        simulations[i] = price_path[-1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(simulations, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=np.percentile(simulations, 5), color='red', linestyle='--', label='5th Percentile')
    plt.axvline(x=np.percentile(simulations, 95), color='red', linestyle='--', label='95th Percentile')
    plt.title(f"Monte Carlo Simulation for {symbol} Price Prediction")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    # Provide textual advice based on simulation
    mean_simulation = np.mean(simulations)
    print(f"Monte Carlo Simulation for {symbol}:")
    print(f"The simulated price of {symbol} in 5 days has an expected mean value of {mean_simulation:.2f}.")
    print(f"Based on the simulation, there's a 90% chance that the price will be between {np.percentile(simulations, 5):.2f} and {np.percentile(simulations, 95):.2f}.")
    print("Investment advice: If the price falls within this range, it may indicate a stable short-term price movement.")

# Plot forecasts for each stock symbol
def plot_stock_forecasts(stock_symbol, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Price', color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3)
    plt.title(f"5-Day Forecast for {stock_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Investment Advice based on Forecast
def investment_advice(forecast):
    if forecast['yhat_upper'].iloc[-1] > forecast['yhat_lower'].iloc[-1]:
        return "The price is expected to increase in the short term. Consider holding or buying more."
    else:
        return "The price is expected to decrease in the short term. Consider reducing your exposure or selling."

# Main function to execute all features
def main():
    print("Welcome to the Smart Investment Advisor!")
    
    input_symbols = input("Enter stock symbols with suffix (e.g., TATASTEEL.NS, RPOWER.BS): ")
    stock_symbols = [symbol.strip() for symbol in input_symbols.split(",")]
    
    quantities = {}
    invested_amounts = {}
    for symbol in stock_symbols:
        quantities[symbol] = int(input(f"Enter quantity for {symbol}: "))
        invested_amounts[symbol] = float(input(f"Enter total amount invested in {symbol}: "))

    stock_data = get_stock_data(stock_symbols)
    index_data = get_index_data()

    index_comparison = compare_with_index(stock_data, index_data)
    print("\nStock Performance Compared to Market Index:")
    for symbol, comparison in index_comparison.items():
        print(f"{symbol}: Stock Return = {comparison['Stock Return']:.2%}, Index Return = {comparison['Index Return']:.2%}, Alpha = {comparison['Alpha']:.2%}")

    portfolio_return, portfolio_risk, sharpe_ratio = portfolio_allocation(stock_data, quantities, invested_amounts)
    print(f"\nPortfolio Return: {portfolio_return:.2%}, Portfolio Risk: {portfolio_risk:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")

    profit_loss = calculate_profit_loss(stock_data, quantities, invested_amounts)
    print("\nProfit/Loss Calculations:")
    for symbol, data in profit_loss.items():
        print(f"{symbol}: Current Price = {data['Current Price']}, Invested Value = {data['Invested Value']}, "
              f"Current Value = {data['Current Value']}, Profit/Loss = {data['Profit/Loss']:.2f} ({data['Profit/Loss %']:.2f}%)")

    print("\nSentiment Analysis:")
    for symbol in stock_symbols:
        sentiment = analyze_sentiment(symbol)
        print(f"Sentiment for {symbol}: {sentiment}")

    print("\n5-Day Price Forecasts:")
    for symbol in stock_symbols:
        forecast = forecast_stock_prices(symbol)
        plot_stock_forecasts(symbol, forecast)
        print(f"Forecast Summary for {symbol}:")
        print(investment_advice(forecast))

    print("\nMonte Carlo Simulations:")
    for symbol in stock_symbols:
        monte_carlo_simulation(stock_data, symbol)

if __name__ == "__main__":
    main()
