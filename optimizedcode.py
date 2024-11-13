import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime
import scipy.stats as stats

# Function to fetch and clean stock data for Prophet forecasting
def get_stock_data_for_forecast(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1y")[['Close']].dropna().reset_index()
    data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)  # Remove timezone if present
    return data

# Forecast stock prices using Prophet
def forecast_stock_prices(stock_symbol):
    data = get_stock_data_for_forecast(stock_symbol)
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast

# Display forecasted prices for the next 5 days
def display_forecast(stock_symbol, forecast):
    print(f"\nPredicted stock prices for {stock_symbol} over the next 5 days:")
    for i in range(1, 6):
        print(f"Day {i}: ₹{forecast['yhat'].iloc[-i]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label="Confidence Interval")
    plt.title(f"Forecast for {stock_symbol} Stock Price (Next 5 Days)")
    plt.xlabel('Date')
    plt.ylabel('Stock Price (₹)')
    plt.legend()
    plt.show()

# Monte Carlo simulation for price prediction
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
    plt.xlabel("Price (₹)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    mean_simulation = np.mean(simulations)
    print(f"\nMonte Carlo Simulation for {symbol}:")
    print(f"Expected mean price in 5 days: ₹{mean_simulation:.2f}")
    print(f"90% chance that the price will be between ₹{np.percentile(simulations, 5):.2f} and ₹{np.percentile(simulations, 95):.2f}.")
    print("\nInvestment insight: If the forecasted price range is close to your target price, this might be a stable short-term investment.")

# News sentiment analysis for stock
def get_news_sentiment(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    news = stock.news
    sentiments = []
    
    for item in news:
        text = item.get('title', '') + " " + item.get('summary', '')
        if text.strip():
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
    
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    sentiment_label = "Neutral"
    if avg_sentiment > 0:
        sentiment_label = "Positive"
    elif avg_sentiment < 0:
        sentiment_label = "Negative"
    
    return sentiment_label

# Calculate portfolio risk management metrics (volatility, VaR, CVaR)
def calculate_portfolio_risk(stock_data, portfolio, confidence_level=0.95):
    aligned_data = []
    for symbol, details in portfolio.items():
        stock = stock_data[symbol]['Close'].pct_change().dropna()
        stock = stock.reindex(stock_data[next(iter(stock_data))].index).fillna(method="ffill").fillna(method="bfill")
        weight = details['total_invested'] / sum([p['total_invested'] for p in portfolio.values()])
        aligned_data.append(stock * weight)
    
    portfolio_returns = pd.DataFrame(aligned_data).T.dropna()
    portfolio_returns['Total'] = portfolio_returns.sum(axis=1)
    portfolio_volatility = np.std(portfolio_returns['Total']) * np.sqrt(252)
    portfolio_var = np.percentile(portfolio_returns['Total'], (1 - confidence_level) * 100)
    portfolio_cvar = portfolio_returns[portfolio_returns['Total'] < portfolio_var]['Total'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns['Total'], bins=50, alpha=0.7, color='green')
    plt.axvline(x=portfolio_var, color='red', linestyle='--', label='VaR (95% Confidence)')
    plt.title("Portfolio Returns Distribution and Value at Risk")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    return portfolio_volatility, portfolio_var, portfolio_cvar

# Generate investment advice based on stock forecast
def investment_advice(symbol, forecast, current_price, quantity):
    predicted_price = forecast['yhat'].iloc[-1]
    price_change = (predicted_price - current_price) / current_price * 100
    
    if price_change > 2:
        recommendation = f"Buy {quantity} shares of {symbol}. Expected price increase: {price_change:.2f}%."
        advice = "The stock shows a strong positive outlook for the near term."
    elif price_change < -2:
        recommendation = f"Sell {quantity} shares of {symbol}. Expected price decrease: {price_change:.2f}%."
        advice = "The stock may decline soon; consider selling to avoid potential losses."
    else:
        recommendation = f"Hold {quantity} shares of {symbol}. The price is expected to stay stable."
        advice = "The stock is likely stable, with no significant fluctuations expected in the short term."
    
    return recommendation + "\nInvestment insight: " + advice

# Portfolio analysis function
def analyze_portfolio(stock_symbols, portfolio, stock_data):
    print("\n--- Portfolio Analysis Report ---\n")
    
    for symbol in stock_symbols:
        forecast = forecast_stock_prices(symbol)
        current_price = yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1]
        
        print(f"\n--- Analysis for {symbol} ---")
        display_forecast(symbol, forecast)
        
        monte_carlo_simulation(stock_data, symbol)
        sentiment = get_news_sentiment(symbol)
        print(f"News Sentiment for {symbol}: {sentiment}")
        
        # Generate investment advice
        quantity = portfolio[symbol]['quantity']
        advice = investment_advice(symbol, forecast, current_price, quantity)
        print(advice)
        
        # Sharpe Ratio, Max Drawdown, and CAGR
        daily_returns = stock_data[symbol]['Close'].pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        max_drawdown = calculate_max_drawdown(stock_data[symbol]['Close'])
        cagr = calculate_cagr(stock_data[symbol]['Close'])
        
        print(f"Sharpe Ratio for {symbol}: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown for {symbol}: {max_drawdown:.2f}")
        print(f"CAGR for {symbol}: {cagr*100:.2f}%")
        
    # Risk and performance analysis
    portfolio_volatility, portfolio_var, portfolio_cvar = calculate_portfolio_risk(stock_data, portfolio)
    print("\n--- Portfolio Risk and Performance Metrics ---")
    print(f"Portfolio Volatility (Annualized): {portfolio_volatility:.2f}")
    print(f"Portfolio Value at Risk (95% Confidence): {portfolio_var:.2f}")
    print(f"Portfolio Conditional VaR: {portfolio_cvar:.2f}")

def main():
    portfolio = {}
    stock_symbols = []
    num_stocks = int(input("Enter the number of stocks in your portfolio: "))
    
    for _ in range(num_stocks):
        symbol = input("Enter stock symbol (e.g., 'RELIANCE.NS'): ").upper()
        quantity = int(input(f"Enter quantity of {symbol} stocks: "))
        amount_invested = float(input(f"Enter total amount invested in {symbol} (₹): "))
        
        stock_symbols.append(symbol)
        portfolio[symbol] = {'quantity': quantity, 'total_invested': amount_invested}
    
    stock_data = {symbol: yf.download(symbol, period="1y") for symbol in stock_symbols}
    analyze_portfolio(stock_symbols, portfolio, stock_data)

if __name__ == "__main__":
    main()
