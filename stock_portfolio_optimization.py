import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from GoogleNews import GoogleNews
from newspaper import Article
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from scipy.optimize import minimize
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Function to fetch historical stock data
def get_stock_data(stock_symbols):
    stock_data = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # 1 year of historical data
        stock_data[symbol] = data
    return stock_data

# Function to fetch news headlines
def fetch_news(stock_symbols):
    news_data = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for symbol in stock_symbols:
        query = f"{symbol} stock news"
        url = f"https://www.google.com/search?q={query}&tbm=nws"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
            headline = item.get_text()
            headlines.append(headline)
        news_data[symbol] = headlines
    return news_data

# Function to clean and preprocess news headlines
def clean_headlines(headlines):
    cleaned_headlines = []
    for headline in headlines:
        tokens = headline.lower().split()
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        cleaned_headlines.append(" ".join(tokens))
    return cleaned_headlines

# Perform sentiment analysis on news headlines
def analyze_sentiment(news_data):
    sentiment_results = {}
    for symbol, headlines in news_data.items():
        cleaned_headlines = clean_headlines(headlines)
        sentiment_scores = []
        for headline in cleaned_headlines:
            score = sia.polarity_scores(headline)['compound']  # Compound sentiment score
            sentiment_scores.append(score)
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        sentiment_results[symbol] = avg_sentiment
    return sentiment_results

# Calculate risk metrics for each stock
def calculate_risk_metrics(stock_data):
    risk_metrics = {}
    for symbol, data in stock_data.items():
        data['Daily Return'] = data['Close'].pct_change()
        volatility = data['Daily Return'].std() * np.sqrt(252)
        running_max = data['Close'].cummax()
        drawdown = (data['Close'] - running_max) / running_max
        max_drawdown = drawdown.min()
        sharpe_ratio = data['Daily Return'].mean() / data['Daily Return'].std() * np.sqrt(252)
        risk_metrics[symbol] = {'Volatility': volatility, 'Max Drawdown': max_drawdown, 'Sharpe Ratio': sharpe_ratio}
    return risk_metrics

# Function to optimize portfolio allocation
def optimize_portfolio(stock_data, quantities, purchase_prices):
    returns = pd.DataFrame()
    for symbol in stock_data:
        data = stock_data[symbol]
        daily_returns = data['Close'].pct_change()
        returns[symbol] = daily_returns
    avg_returns = returns.mean()
    cov_matrix = returns.cov()
    def objective(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_volatility
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(stock_data)))
    init_guess = [1. / len(stock_data)] * len(stock_data)
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    optimal_weights = result.x
    portfolio_value = sum(optimal_weights[i] * purchase_prices[stock] * quantities[stock] for i, stock in enumerate(stock_data))
    return optimal_weights, portfolio_value

# Plotting portfolio performance over time
def plot_portfolio_performance(stock_data):
    plt.figure(figsize=(10, 6))
    for symbol, data in stock_data.items():
        data['Daily Return'] = data['Close'].pct_change()
        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod() - 1
        plt.plot(data.index, data['Cumulative Return'], label=f"{symbol}", linestyle='--')
    plt.title("Portfolio Performance Comparison Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Generating investment advice based on portfolio & sentiment
def generate_investment_advice(optimal_weights, sentiment_results, risk_metrics):
    advice = []
    for i, symbol in enumerate(sentiment_results.keys()):
        weight = optimal_weights[i]
        sentiment = sentiment_results[symbol]
        risk = risk_metrics[symbol]
        sentiment_advice = "Hold" if -0.05 < sentiment < 0.05 else ("Buy" if sentiment > 0 else "Sell")
        risk_advice = f"{symbol} has a volatility of {risk['Volatility']:.2f} and a Sharpe Ratio of {risk['Sharpe Ratio']:.2f}. " \
                      f"Consider {'increasing' if weight < 0.1 else 'maintaining' if 0.1 <= weight < 0.2 else 'reducing'} exposure."
        advice.append(f"Stock: {symbol}, Sentiment Score: {sentiment:.2f} ({sentiment_advice}), {risk_advice}")
    return advice

# Main function to execute all steps
def main():
    input_symbols = input("Enter stock symbols with suffix (e.g., TATASTEEL.NS, RPOWER.BS): ")
    stock_symbols = [symbol.strip() for symbol in input_symbols.split(",")]
    quantities = {}
    purchase_prices = {}
    for symbol in stock_symbols:
        quantities[symbol] = int(input(f"Enter quantity for {symbol}: "))
        purchase_prices[symbol] = float(input(f"Enter purchase price for {symbol}: "))
    
    stock_data = get_stock_data(stock_symbols)
    plot_portfolio_performance(stock_data)
    risk_metrics = calculate_risk_metrics(stock_data)
    optimal_weights, portfolio_value = optimize_portfolio(stock_data, quantities, purchase_prices)
    news_data = fetch_news(stock_symbols)
    sentiment_results = analyze_sentiment(news_data)
    
    print(f"\nOptimized Portfolio Allocation: {optimal_weights}")
    print(f"Optimized Portfolio Value: {portfolio_value:.2f} INR\n")
    
    investment_advice = generate_investment_advice(optimal_weights, sentiment_results, risk_metrics)
    print("\nInvestment Advice:")
    for line in investment_advice:
        print(line)

if __name__ == "__main__":
    main()
