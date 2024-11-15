import yfinance as yf
from nsepy import get_history
from datetime import date
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from prophet import Prophet
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def get_stock_data(stock_symbols):
    stock_data = {}
    for symbol in stock_symbols:
        try:
            stock_data[symbol] = yf.download(symbol, start="2020-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return stock_data

# **1. Stop-Loss Calculation**
def calculate_stop_loss(price, percentage):
    return price * (1 - percentage / 100)

# **2. Sector Allocation**
def get_sector_data(portfolio):
    sector_allocation = {}
    for stock in portfolio:
        try:
            symbol = stock['symbol']
            stock = yf.Ticker(symbol)
            sector = stock.info.get('sector', 'Unknown')
            
            if sector in sector_allocation:
                sector_allocation[sector] += 1
            else:
                sector_allocation[sector] = 1
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return sector_allocation

def plot_sector_allocation(sector_allocation):
    # Plot sector allocation using pie chart
    sectors = list(sector_allocation.keys())
    values = list(sector_allocation.values())

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=sectors, autopct='%1.1f%%', startangle=140)
    plt.title('Portfolio Sector Allocation')
    plt.show()


def get_stock_sector(symbol):
    # Fetch sector from yfinance or a predefined dictionary (for demo purposes)
    stock_info = yf.Ticker(symbol).info
    return stock_info.get('sector', 'Unknown')

# **3. Portfolio Performance vs. Index (NIFTY 50)**
def portfolio_vs_nifty(stock_data, portfolio):
    # Get NIFTY 50 data using yfinance
    nifty_data = yf.download("^NSEI", start="2023-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    nifty_returns = nifty_data['Adj Close'].pct_change().cumsum()

    portfolio_returns = []

    for stock in portfolio:
        symbol = stock['symbol']
        price_history = stock_data[symbol]['Adj Close']
        stock_returns = price_history.pct_change().cumsum()
        portfolio_returns.append(stock_returns)

    portfolio_returns = pd.DataFrame(portfolio_returns).T
    portfolio_returns.columns = [stock['symbol'] for stock in portfolio]

    # Calculate portfolio performance
    portfolio_performance = portfolio_returns.mean(axis=1)
    
    # Plot the portfolio vs NIFTY
    plt.figure(figsize=(10, 6))
    plt.plot(nifty_returns, label='NIFTY 50', color='blue')
    plt.plot(portfolio_performance, label='Portfolio', color='green')
    plt.title("Portfolio vs NIFTY 50 Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()
# **4. Current Portfolio Value**
def current_portfolio_value(portfolio, stock_data):
    return sum(
        stock_data[stock['symbol']]['Adj Close'].iloc[-1] * stock['quantity']
        for stock in portfolio
    )

# **5. Predicted Portfolio Value (Monte Carlo Simulation)**
def monte_carlo_simulation(stock_data, portfolio, num_simulations=1000, num_days=252):
    simulations = []
    
    for stock in portfolio:
        symbol = stock['symbol']
        price_history = stock_data[symbol]['Adj Close']
        log_returns = np.log(price_history / price_history.shift(1)).dropna()

        mean = log_returns.mean()
        std_dev = log_returns.std()

        # Monte Carlo Simulation
        simulated_prices = []
        for _ in range(num_simulations):
            price_simulation = [price_history.iloc[-1]]
            for _ in range(num_days):
                price_simulation.append(price_simulation[-1] * np.exp(np.random.normal(mean, std_dev)))
            simulations.append(price_simulation)
    
    # Convert simulations to numpy array for better handling
    simulations = np.array(simulations)
    
    # Plotting Monte Carlo Simulation - Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(simulations[:, -1], bins=50, edgecolor='k', alpha=0.7)
    plt.title("Monte Carlo Simulation - Future Stock Price Distribution")
    plt.xlabel("Stock Price")
    plt.ylabel("Frequency")
    plt.show()
    
    return simulations
# **6. Trending Sectors**
def trending_sectors():
    url = "https://www.moneycontrol.com/stocks/marketstats/index.php"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    sectors = soup.find_all('a', class_='statslinks')
    return [sector.text for sector in sectors[:5]]

# **7. Stock Recommendations**
def stock_recommendations(portfolio, stock_data):
    recommendations = []
    
    for stock in portfolio:
        symbol = stock['symbol']
        sector = get_stock_sector(symbol)
        performance = stock_data[symbol]['Adj Close'].pct_change().tail(1).iloc[0]  # Last day performance

        # Basic logic for recommendations (for demonstration purposes)
        if performance > 0.05:
            recommendation = 'Buy'
        elif performance < -0.05:
            recommendation = 'Sell'
        else:
            recommendation = 'Hold'
        
        recommendations.append({
            'symbol': symbol,
            'sector': sector,
            'performance': performance,
            'recommendation': recommendation
        })
    
    return recommendations


# **8. Investment Advice with Sentiment**
def sentiment_analysis(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock+market"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    articles = soup.find_all('item')
    
    sentiment_scores = []
    for article in articles:
        text = article.title.text + " " + article.description.text
        analysis = TextBlob(text).sentiment
        sentiment_scores.append(analysis.polarity)
    
    avg_sentiment = np.mean(sentiment_scores)
    sentiment = "Positive" if avg_sentiment > 0.1 else "Neutral" if avg_sentiment > -0.1 else "Negative"
    return sentiment

# **9. Profit/Loss Calculation**
def profit_loss(portfolio, stock_data):
    results = []
    for stock in portfolio:
        current_price = stock_data[stock['symbol']]['Adj Close'].iloc[-1]
        total_investment = stock['investment']
        profit_or_loss = (current_price * stock['quantity']) - total_investment
        results.append({'symbol': stock['symbol'], 'P/L': profit_or_loss})
    return results
# **10. Portfolio Optimization**
def portfolio_optimization(stock_data, portfolio):
    # If there's only one stock, no optimization needed
    if len(portfolio) == 1:
        return {portfolio[0]['symbol']: 1.0}  # Assign 100% weight to the single stock

    prices = pd.DataFrame({stock['symbol']: stock_data[stock['symbol']]['Adj Close'] for stock in portfolio})
    mean_returns = expected_returns.mean_historical_return(prices)
    covariance = risk_models.sample_cov(prices)
    
    ef = EfficientFrontier(mean_returns, covariance)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    ef.portfolio_performance(verbose=True)
    return cleaned_weights

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_candlestick(df):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3)
    fig.add_trace(go.Candlestick(
        x=df['ds'],
        open=df['yhat_lower'],
        high=df['yhat'],
        low=df['yhat_lower'],
        close=df['yhat_upper'],
        name='Stock Price Prediction',
    ))

    fig.update_layout(title='Stock Price Prediction - Candlestick Chart', xaxis_rangeslider_visible=False)
    fig.show()


# **12. Stock Price Prediction using Prophet**
def predict_stock_price(symbol, stock_data):
    data = stock_data[['Close']].reset_index()
    data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    data['ds'] = data['ds'].dt.tz_localize(None)

    model = Prophet()
    model.fit(data)
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)
    # Plotting Prophet prediction using Candlestick chart
    plot_candlestick(forecast)
# **13. Calculate Technical Indicators**
def calculate_technical_indicators(stock_data, symbol):
    # Moving Average Convergence Divergence (MACD)
    stock_data[symbol]['MACD'] = stock_data[symbol]['Adj Close'].ewm(span=12, adjust=False).mean() - stock_data[symbol]['Adj Close'].ewm(span=26, adjust=False).mean()
    stock_data[symbol]['Signal'] = stock_data[symbol]['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = stock_data[symbol]['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data[symbol]['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    stock_data[symbol]['Moving_Avg'] = stock_data[symbol]['Adj Close'].rolling(window=20).mean()
    stock_data[symbol]['Upper_Band'] = stock_data[symbol]['Moving_Avg'] + (2 * stock_data[symbol]['Adj Close'].rolling(window=20).std())
    stock_data[symbol]['Lower_Band'] = stock_data[symbol]['Moving_Avg'] - (2 * stock_data[symbol]['Adj Close'].rolling(window=20).std())

    return stock_data[symbol][['MACD', 'Signal', 'RSI', 'Upper_Band', 'Lower_Band']].tail()

# **14. News Sentiment Analysis using Google RSS Feed**
def news_sentiment_analysis(stock_symbols):
    news_sentiments = {}
    for symbol in stock_symbols:
        sentiment = sentiment_analysis(symbol)
        news_sentiments[symbol] = sentiment
    return news_sentiments

# **15. Detailed Summary of the User's Portfolio**
def portfolio_summary(portfolio, stock_data):
    summary = []
    total_value = 0
    for stock in portfolio:
        current_price = stock_data[stock['symbol']]['Adj Close'].iloc[-1]
        stock_value = current_price * stock['quantity']
        total_value += stock_value
        summary.append({
            'symbol': stock['symbol'],
            'quantity': stock['quantity'],
            'current_price': current_price,
            'total_value': stock_value,
            'investment': stock['investment'],
            'P/L': stock_value - stock['investment']
        })
    return summary, total_value

# **16. Portfolio Potential Analysis**
def portfolio_potential_analysis(portfolio, stock_data):
    potential_values = []
    for stock in portfolio:
        symbol = stock['symbol']
        future_prices = monte_carlo_simulation(stock_data, portfolio)
        expected_value = np.mean(future_prices) * stock['quantity']
        potential_values.append({
            'symbol': symbol,
            'expected_value': expected_value
        })
    return potential_values
# **User Input Handling**
def get_user_input():
    portfolio = []
    while True:
        symbol = input("Enter stock symbol (or type 'done' to finish): ")
        if symbol.lower() == 'done':
            break
        quantity = int(input(f"Enter quantity for {symbol}: "))
        investment = float(input(f"Enter amount invested in {symbol}: "))
        portfolio.append({'symbol': symbol, 'quantity': quantity, 'investment': investment})
    return portfolio

# **Final Report Generation**
def generate_report(stock_data, portfolio):
    print("Generating report...\n")
    
    # 1. Sector Allocation
    print("Sector Allocation:")
    sector_allocation = get_sector_data(portfolio)
    plot_sector_allocation(sector_allocation)
    
    # 2. Portfolio Performance vs NIFTY
    print("Portfolio vs NIFTY 50 Performance:")
    portfolio_vs_nifty(stock_data, portfolio)
    
    # 3. Current Portfolio Value
    portfolio_value = current_portfolio_value(portfolio, stock_data)
    print(f"Current Portfolio Value: ₹{portfolio_value:,.2f}")
    
    # 4. Predicted Portfolio Value (Monte Carlo Simulation)
    print("Monte Carlo Simulation Results:")
    monte_carlo_results = monte_carlo_simulation(stock_data, portfolio)
    
    # 5. Stock Recommendations
    print("Stock Recommendations:")
    recommendations = stock_recommendations(portfolio, stock_data)
    for recommendation in recommendations:
        print(f"{recommendation['symbol']} - {recommendation['recommendation']}")

    # 6. Sentiment Analysis
    print("Sentiment Analysis:")
    news_sentiments = news_sentiment_analysis([stock['symbol'] for stock in portfolio])
    for symbol, sentiment in news_sentiments.items():
        print(f"{symbol}: {sentiment}")
    
    # 7. Profit/Loss Calculation
    print("Profit/Loss:")
    profit_loss_results = profit_loss(portfolio, stock_data)
    for result in profit_loss_results:
        print(f"{result['symbol']} - P/L: ₹{result['P/L']:,.2f}")
    
    # 8. Portfolio Optimization
    print("Optimizing Portfolio:")
    optimized_weights = portfolio_optimization(stock_data, portfolio)
    print("Optimized Weights:")
    for symbol, weight in optimized_weights.items():
        print(f"{symbol}: {weight * 100:.2f}%")
    
    # 9. Technical Indicators
    print("Technical Indicators:")
    for stock in portfolio:
        print(f"\n{stock['symbol']} Technical Indicators:")
        technical_indicators = calculate_technical_indicators(stock_data, stock['symbol'])
        print(technical_indicators)
    
    # 10. Stock Price Prediction (Prophet)
    for stock in portfolio:
        print(f"\n{stock['symbol']} Stock Price Prediction:")
        predict_stock_price(stock['symbol'], stock_data[stock['symbol']])
    
    # 11. Detailed Summary of Portfolio
    print("\nDetailed Portfolio Summary:")
    summary, total_value = portfolio_summary(portfolio, stock_data)
    for item in summary:
        print(f"{item['symbol']}: Quantity {item['quantity']}, Total Value: ₹{item['total_value']:,.2f}, P/L: ₹{item['P/L']:,.2f}")
    print(f"Total Portfolio Value: ₹{total_value:,.2f}")
    
    # 12. Portfolio Potential
    print("\nPortfolio Potential Analysis:")
    potential_values = portfolio_potential_analysis(portfolio, stock_data)
    for value in potential_values:
        print(f"{value['symbol']} expected value: ₹{value['expected_value']:,.2f}")
    
    print("\nReport generation complete!")

# Main Execution
if __name__ == "__main__":
    portfolio = get_user_input()
    stock_symbols = [stock['symbol'] for stock in portfolio]
    stock_data = get_stock_data(stock_symbols)
    
    # Generate report
    generate_report(stock_data, portfolio)
