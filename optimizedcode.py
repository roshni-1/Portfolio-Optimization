import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from textblob import TextBlob
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from datetime import datetime
import requests
import seaborn as sns

# Set up for better visualization
plt.style.use('seaborn-v0_8-darkgrid')

# Function to get stock data from Yahoo Finance
def get_stock_data(stock_symbols):
    stock_data = {}
    for symbol in stock_symbols:
        data = yf.download(symbol + ".NS", period="1y")
        if not data.empty:
            stock_data[symbol] = data
    return stock_data

# Profit and Loss Calculation
def calculate_profit_loss(stock_data, portfolio):
    total_invested = 0
    total_current_value = 0
    profit_loss_details = []

    for symbol, details in portfolio.items():
        if symbol in stock_data:
            last_price = stock_data[symbol]['Close'].iloc[-1]
            invested = details['total_invested']
            current_value = last_price * details['quantity']
            profit_loss = current_value - invested
            total_invested += invested
            total_current_value += current_value
            profit_loss_details.append(
                {
                    "symbol": symbol,
                    "invested": invested,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "percent_change": (profit_loss / invested) * 100
                }
            )

    total_profit_loss = total_current_value - total_invested
    return total_invested, total_current_value, total_profit_loss, profit_loss_details

# Stop-Loss Calculation (based on user-defined percentage)
def stop_loss_calculation(stock_data, portfolio, stop_loss_percent=5):
    stop_loss_details = {}
    for symbol, details in portfolio.items():
        last_price = stock_data[symbol]['Close'].iloc[-1]
        stop_loss_price = last_price * (1 - stop_loss_percent / 100)
        stop_loss_details[symbol] = {
            "current_price": last_price,
            "stop_loss_price": stop_loss_price,
            "threshold": stop_loss_percent
        }
    return stop_loss_details

# Sector Allocation of the Portfolio
def sector_allocation(stock_symbols):
    sectors = {}
    for symbol in stock_symbols:
        info = yf.Ticker(symbol + ".NS").info
        sector = info.get('sector', 'Unknown')
        if sector in sectors:
            sectors[sector] += 1
        else:
            sectors[sector] = 1
    return sectors

# Portfolio Performance vs Index (NIFTY 50)
def portfolio_vs_index(stock_data, portfolio, index_symbol="^NSEI"):
    index_data = yf.download(index_symbol, period="1y")['Close']
    portfolio_returns = []
    
    for symbol, details in portfolio.items():
        data = stock_data[symbol]['Close']
        returns = data.pct_change().fillna(0)
        weighted_return = returns * (details['quantity'] / sum([p['quantity'] for p in portfolio.values()]))
        portfolio_returns.append(weighted_return)
    
    total_portfolio_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
    total_portfolio_returns.index = pd.to_datetime(total_portfolio_returns.index)
    index_returns = index_data.pct_change().fillna(0)
    
    return total_portfolio_returns, index_returns

# News Sentiment Analysis Using Google News RSS Feed
def news_sentiment_analysis(stock_symbols):
    sentiment_scores = {}
    for symbol in stock_symbols:
        news_url = f"https://news.google.com/rss/search?q={symbol}"
        response = requests.get(news_url)
        news_feed = response.text
        
        sentiment_score = 0
        if news_feed:
            news_items = news_feed.split("<item>")
            for item in news_items[1:]:
                title = item.split("<title>")[1].split("</title>")[0]
                blob = TextBlob(title)
                sentiment_score += blob.sentiment.polarity
        sentiment_scores[symbol] = sentiment_score
    return sentiment_scores

# Portfolio Optimization Using Sharpe Ratio
def optimize_portfolio(stock_data, portfolio, risk_free_rate=0.03):
    prices_df = pd.DataFrame({symbol: data['Close'] for symbol, data in stock_data.items()})
    mu = mean_historical_return(prices_df)
    S = CovarianceShrinkage(prices_df).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, performance

# Monte Carlo Simulation for Price Prediction
def monte_carlo_simulation(data, num_simulations=1000, time_horizon=30):
    daily_returns = data['Close'].pct_change().dropna()
    last_price = data['Close'].iloc[-1]

    mean_return = daily_returns.mean()
    volatility = daily_returns.std()

    simulation_results = np.zeros((time_horizon, num_simulations))
    for sim in range(num_simulations):
        prices = [last_price]
        for day in range(time_horizon):
            simulated_price = prices[-1] * (1 + np.random.normal(mean_return, volatility))
            prices.append(simulated_price)
        simulation_results[:, sim] = prices[1:]

    mean_price = np.mean(simulation_results[-1, :])
    fifth_percentile = np.percentile(simulation_results[-1, :], 5)
    ninety_fifth_percentile = np.percentile(simulation_results[-1, :], 95)

    return mean_price, fifth_percentile, ninety_fifth_percentile, simulation_results

# Candlestick Chart using Plotly
def plot_candlestick(stock_symbol, stock_data):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=stock_symbol
    )])

    fig.update_layout(
        title=f'Candlestick Chart for {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        xaxis_rangeslider_visible=False
    )
    fig.show()

# Stock Price Prediction using Prophet
def predict_stock_price(stock_symbol, stock_data):
    data = stock_data[['Close']].reset_index()
    data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)

    model = Prophet()
    model.fit(data)
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label="Actual Prices")
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecasted Prices", color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
    plt.title(f"Price Forecast for {symbol}")
    plt.legend()
    plt.show()

    return forecast
# Generate the Full Report with Detailed Explanations
def generate_report(stock_data, portfolio):
    # Calculate profit/loss
    total_invested, total_current_value, total_profit_loss, profit_loss_details = calculate_profit_loss(stock_data, portfolio)
    
    # Stop-loss Calculation
    stop_loss_details = stop_loss_calculation(stock_data, portfolio)
    
    # Sector Allocation
    sectors = sector_allocation(list(portfolio.keys()))
    
    # Portfolio Performance vs Index
    portfolio_returns, index_returns = portfolio_vs_index(stock_data, portfolio)
    
    # News Sentiment Analysis
    sentiment_scores = news_sentiment_analysis(list(portfolio.keys()))
    
    # Portfolio Optimization
    cleaned_weights, performance = optimize_portfolio(stock_data, portfolio)
    
    # Monte Carlo Simulation
    monte_carlo_results = {}
    for symbol, data in stock_data.items():
        mean_price, fifth_percentile, ninety_fifth_percentile, simulation_results = monte_carlo_simulation(data)
        monte_carlo_results[symbol] = {
            'mean': mean_price,
            '5th_percentile': fifth_percentile,
            '95th_percentile': ninety_fifth_percentile
        }
        # Plot Monte Carlo Simulation
        plt.figure(figsize=(10,6))
        plt.plot(simulation_results, color='blue', alpha=0.1)
        plt.title(f"Monte Carlo Simulation for {symbol}")
        plt.xlabel('Days')
        plt.ylabel('Price (INR)')
        plt.show()

    # Plot Candlestick Charts and Stock Price Predictions
    for symbol in portfolio.keys():
        plot_candlestick(symbol, stock_data[symbol])
        predict_stock_price(symbol, stock_data[symbol])
    
    # Print Detailed Report
    print("\n--- Portfolio Summary ---")
    print(f"Total Invested: ₹{total_invested:.2f}")
    print(f"Total Current Value: ₹{total_current_value:.2f}")
    print(f"Total Profit/Loss: ₹{total_profit_loss:.2f} ({(total_profit_loss/total_invested)*100:.2f}%)")
    
    print("\n--- Sector Allocation ---")
    for sector, count in sectors.items():
        print(f"{sector}: {count} stocks")
    
    print("\n--- Portfolio vs NIFTY 50 ---")
    print(f"Portfolio Return: {portfolio_returns.sum() * 100:.2f}%")
    print(f"NIFTY 50 Return: {index_returns.sum() * 100:.2f}%")
    
    print("\n--- News Sentiment Analysis ---")
    for symbol, sentiment in sentiment_scores.items():
        print(f"{symbol}: Sentiment Score = {sentiment:.2f}")
        if sentiment > 0:
            print(f"Sentiment is positive, which may positively impact the stock.")
        elif sentiment < 0:
            print(f"Sentiment is negative, which may negatively impact the stock.")
        else:
            print(f"Sentiment is neutral.")
    
    print("\n--- Portfolio Optimization (Sharpe Ratio) ---")
    print(f"Optimized Portfolio Weights: {cleaned_weights}")
    print(f"Expected Annual Return: {performance[0] * 100:.2f}%")
    print(f"Annual Volatility: {performance[1] * 100:.2f}%")
    print(f"Sharpe Ratio: {performance[2]:.2f}")
    
    print("\n--- Monte Carlo Simulations ---")
    for symbol, result in monte_carlo_results.items():
        print(f"{symbol}: Predicted 30-day Price Range = ₹{result['5th_percentile']:.2f} to ₹{result['95th_percentile']:.2f} (Mean: ₹{result['mean']:.2f})")

# User Input for stock symbol, quantity, total amount invested
portfolio = {}
print("Enter your portfolio details:")
while True:
    symbol = input("Enter stock symbol (or type 'done' to finish): ").upper()
    if symbol.lower() == 'done':
        break
    quantity = int(input(f"Enter quantity for {symbol}: "))
    invested = float(input(f"Enter amount invested in {symbol}: ₹"))
    portfolio[symbol] = {'quantity': quantity, 'total_invested': invested}

# Fetch stock data
stock_symbols = list(portfolio.keys())
stock_data = get_stock_data(stock_symbols)

# Generate and display the report
generate_report(stock_data, portfolio)
