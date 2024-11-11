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
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Function to fetch historical stock data
def get_stock_data(stock_symbols):
    stock_data = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")  # Fetch all available historical data
        data.index = data.index.tz_localize(None) 
    return stock_data

# Fetch news headlines
def fetch_news(stock_symbols):
    news_data = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for symbol in stock_symbols:
        query = f"{symbol} stock news"
        url = f"https://www.google.com/search?q={query}&tbm=nws"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [item.get_text() for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')]
        news_data[symbol] = headlines
    return news_data

# Clean & preprocess news headlines
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
        sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in cleaned_headlines]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        sentiment_results[symbol] = avg_sentiment
    return sentiment_results

# Function to fetch current stock prices
def get_current_price(stock_symbols):
    current_prices = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            current_prices[symbol] = data['Close'].iloc[-1]
        else:
            current_prices[symbol] = None  # Handle cases where no price data is found
    return current_prices

# Function to calculate profit and loss
def calculate_profit_loss(stock_symbols, quantities, invested_amounts):
    current_prices = get_current_price(stock_symbols)
    profit_loss_summary = {}

    for symbol in stock_symbols:
        current_price = current_prices[symbol]
        if current_price is None:
            print(f"Warning: No data available for {symbol}.")
            continue
        
        total_current_value = current_price * quantities[symbol]
        total_investment = invested_amounts[symbol]
        profit_or_loss = total_current_value - total_investment
        advice = ""

        # Determine advice based on profit or loss
        if profit_or_loss > 0:
            advice = (
                "Hold/Sell - Since the stock has appreciated, consider holding if you expect further gains, "
                "or sell to lock in profits. Evaluate based on market trends and financial goals."
            )
        elif profit_or_loss < 0:
            advice = (
                "Buy more - The stock has depreciated, so buying more can help you lower your average cost per share, "
                "especially if you believe the stock will recover and appreciate in the future."
            )
        else:
            advice = (
                "Hold - The stock is at break-even. Monitor its performance and market conditions closely to decide "
                "if it aligns with your financial strategy."
            )

        profit_loss_summary[symbol] = {
            "Total Current Value": total_current_value,
            "Total Investment": total_investment,
            "Profit or Loss": profit_or_loss,
            "Advice": advice
        }

    return profit_loss_summary

# Calculate risk metrics for each stock
def calculate_risk_metrics(stock_data):
    risk_metrics = {}
    for symbol, data in stock_data.items():
        data['Daily Return'] = data['Close'].pct_change()
        volatility = data['Daily Return'].std() * np.sqrt(252)
        max_drawdown = ((data['Close'] - data['Close'].cummax()) / data['Close'].cummax()).min()
        sharpe_ratio = data['Daily Return'].mean() / data['Daily Return'].std() * np.sqrt(252)
        risk_metrics[symbol] = {'Volatility': volatility, 'Max Drawdown': max_drawdown, 'Sharpe Ratio': sharpe_ratio}
    return risk_metrics

# Prophet forecasting function
def prophet_forecast(df, periods=30):
    # Prepare data for Prophet
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)  
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)  
    model.fit(df)
    
    # Generate future predictions
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Optimize portfolio allocation
def optimize_portfolio(stock_data, quantities, purchase_prices):
    returns = pd.DataFrame({symbol: stock_data[symbol]['Close'].pct_change() for symbol in stock_data})
    cov_matrix = returns.cov()
    init_guess = [1. / len(stock_data)] * len(stock_data)
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(len(stock_data)))
    result = minimize(lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))), init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    optimal_weights = result.x
    portfolio_value = sum(optimal_weights[i] * purchase_prices[symbol] * quantities[symbol] for i, symbol in enumerate(stock_data))
    return optimal_weights, portfolio_value

def generate_investment_advice(optimal_weights, sentiment_results, risk_metrics, forecast_results, quantities, purchase_prices):
    advice = []
    for i, symbol in enumerate(sentiment_results.keys()):
        weight = optimal_weights[i]
        sentiment = sentiment_results[symbol]
        risk = risk_metrics[symbol]
        
        # Sentiment-based recommendation
        sentiment_advice = ""
        if sentiment > 0.1:
            sentiment_advice = "Buy: The sentiment is positive, suggesting a potential for price appreciation."
        elif sentiment < -0.1:
            sentiment_advice = "Sell: The sentiment is negative, indicating possible price decline."
        else:
            sentiment_advice = "Hold: Sentiment is neutral, indicating no strong movement expected."
        
        # Extract forecasted price range from Prophet's forecast results
        forecast = forecast_results[symbol]
        
        # Check if forecast is a DataFrame and contains 'yhat_lower' and 'yhat_upper'
        if isinstance(forecast, pd.DataFrame):
            forecasted_price_range = (forecast['yhat_lower'].iloc[-1], forecast['yhat_upper'].iloc[-1])
            forecast_advice = f"Forecasted price range: {forecasted_price_range[0]:.2f} to {forecasted_price_range[1]:.2f} (within 30 days)"
        else:
            forecast_advice = "Forecast data not available."
        
        # Risk-based advice
        risk_advice = ""
        if risk['Volatility'] > 0.5:
            risk_advice = "High risk: The stock has high volatility, consider holding or selling if you're risk-averse."
        elif risk['Sharpe Ratio'] < 1:
            risk_advice = "Low Sharpe ratio: The stock may not provide good risk-adjusted returns, hold with caution."
        else:
            risk_advice = "Good risk-adjusted return: The stock has a good risk profile, making it safer to hold or buy."
        
        # Actionable advice: Buy/Sell/Hold and how many shares
        action_advice = ""
        if sentiment > 0.1:
            action_advice = f"Buy: You should consider purchasing {int(weight * 100)} more shares."
        elif sentiment < -0.1:
            action_advice = f"Sell: Consider selling some of your shares to lock in profits or limit losses."

        advice.append(f"{symbol}: {sentiment_advice} {forecast_advice} {risk_advice} {action_advice}")
    
    return advice



# Main function to execute all steps
def main():
    input_symbols = input("Enter stock symbols with suffix (e.g., TATASTEEL.NS, ADANIPORTS.NS): ")
    stock_symbols = [symbol.strip() for symbol in input_symbols.split(",")]
    quantities = {}
    invested_amounts = {}

    for symbol in stock_symbols:
        quantities[symbol] = int(input(f"Enter quantity for {symbol}: "))
        invested_amounts[symbol] = float(input(f"Enter total amount invested in {symbol}: "))

    # Fetch stock data and perform analysis
    stock_data = get_stock_data(stock_symbols)
    news_data = fetch_news(stock_symbols)
    sentiment_results = analyze_sentiment(news_data)
    risk_metrics = calculate_risk_metrics(stock_data)
    current_prices = get_current_price(stock_symbols)
    profit_loss_summary = calculate_profit_loss(stock_symbols, quantities, invested_amounts)

    # Forecast prices using Prophet
    forecast_results = {}
    for symbol in stock_symbols:
        forecast_results[symbol] = prophet_forecast(stock_data[symbol])

    # Optimize portfolio allocation
    optimal_weights, portfolio_value = optimize_portfolio(stock_data, quantities, invested_amounts)

    # Generate detailed investment advice
    investment_advice = generate_investment_advice(optimal_weights, sentiment_results, risk_metrics, forecast_results, quantities, invested_amounts)

    # Display all results
    print("\nProfit and Loss Summary:")
    for symbol, summary in profit_loss_summary.items():
        print(f"\nStock: {symbol}")
        print(f"  Total Current Value: {summary['Total Current Value']}")
        print(f"  Total Investment: {summary['Total Investment']}")
        print(f"  Profit or Loss: {summary['Profit or Loss']}")
        print(f"  Advice: {summary['Advice']}")
    
    print("\nInvestment Advice Summary:")
    for advice_item in investment_advice:
        print(advice_item)

if __name__ == "__main__":
    main()
