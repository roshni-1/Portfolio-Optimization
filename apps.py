# Importing Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
from plotly import express as px
from scipy.optimize import minimize
import ta
import feedparser
from textblob import TextBlob
from nsetools import Nse
from nsepy.symbols import get_symbol_list
from datetime import timedelta, datetime
import requests
from bs4 import BeautifulSoup
import http.client
import json

# Constants
RISK_FREE_RATE = 7.365 / 100  # For portfolio optimization

st.set_page_config(
    page_title="Portfolio Wizard", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancements
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .section-header {
            color: #0056b3;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .line-divider {
            height: 2px;
            background-color: #cccccc;
            margin: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper Functions ---
def fetch_stock_data(symbol, period="6mo"):
    """Fetch historical stock data using yfinance."""
    try:
        data = yf.download(symbol, period=period, interval="1d")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_nifty_50_symbols():
    """Fetch NIFTY 50 symbols using nsepy."""
    nifty_50_symbols = get_symbol_list()
    return nifty_50_symbols

# Fetch symbols
nifty_50_symbols = fetch_nifty_50_symbols()
print(f"NIFTY 50 Symbols: {nifty_50_symbols}")

def monte_carlo_simulation(data, days=30, simulations=1000):
    """Perform Monte Carlo simulation for stock prices."""
    last_price = data["Close"].iloc[-1]
    returns = data["Close"].pct_change().dropna()
    mean_return = returns.mean()
    std_dev = returns.std()

    simulated_prices = []
    for _ in range(simulations):
        simulated_path = [last_price]
        for _ in range(days):
            price = simulated_path[-1] * (1 + np.random.normal(mean_return, std_dev))
            simulated_path.append(price)
        simulated_prices.append(simulated_path[-1])
    
    return simulated_prices

def predict_prices(data, forecast_days=30):
    """Predicting stock prices using Prophet."""
    df = data.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast

def fetch_google_news(stock_symbol):
    """Fetching news articles related to stock symbol using Google RSS feed."""
    rss_url = f"https://news.google.com/rss/search?q={stock_symbol}&hl=en-IN&gl=IN&ceid=IN:en"
    news_feed = feedparser.parse(rss_url)
    articles = []
    for entry in news_feed.entries[:5]:  # Fetch top 5 articles
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })
    return articles

def optimize_portfolio(returns, risk_free_rate=RISK_FREE_RATE):
    """Optimizing portfolio weights using the Sharpe Ratio."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(neg_sharpe_ratio, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x

def fetch_market_holidays():
    """Fetching non-trading days by checking gaps in historical stock data."""
    try:
        # Downloading historical data for NIFTY 50 index
        historical_data = yf.download("^NSEI", period="1y", interval="1d")
        
        # Extracting all dates from the period and checking for missing dates
        all_dates = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max())
        trading_days = historical_data.index  # Actual trading days
        non_trading_days = set(all_dates) - set(trading_days)
        
        # Returning non-trading days as strings
        return set(non_trading_day.strftime('%Y-%m-%d') for non_trading_day in non_trading_days)
    except Exception as e:
        st.error(f"Error fetching market holidays: {e}")
        return set()  # Returning an empty set on failure

def get_next_trading_days(start_date, num_days=5):
    """Fetching the next 'num_days' trading days starting from given date."""
    holidays = fetch_market_holidays()  # Dynamically fetched holidays
    trading_days = []
    current_date = start_date
    while len(trading_days) < num_days:
        # Skip weekends and dynamically fetched holidays
        if current_date.weekday() < 5 and current_date.strftime('%Y-%m-%d') not in holidays:
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    return trading_days

def calculate_volume_oscillator(data, fast_window=12, slow_window=26):
    """Calculate Volume Oscillator."""
    fast_ema = data["Volume"].ewm(span=fast_window, adjust=False).mean()
    slow_ema = data["Volume"].ewm(span=slow_window, adjust=False).mean()
    volume_oscillator = ((fast_ema - slow_ema) / slow_ema) * 100
    return volume_oscillator
# --- Functions to Fetch Market Movers ---
def fetch_market_movers(region="IN", lang="en-US", start=0, count=10):
    """Fetch market movers (top gainers, losers, active stocks) using Yahoo Finance API."""
    conn = http.client.HTTPSConnection("apidojo-yahoo-finance-v1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5d63bb22bemshb6e582f5cdfd2cdp1d4344jsn14c1f5b16633",
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    try:
        # Requesting market movers
        conn.request("GET", f"/market/v2/get-movers?region={region}&lang={lang}&start={start}&count={count}", headers=headers)
        res = conn.getresponse()
        data = res.read()

        # Decoding and parsing response
        response_json = json.loads(data.decode("utf-8"))

        # Extracting movers data
        movers = []
        for category in response_json.get("finance", {}).get("result", []):
            category_title = category.get("title", "Unknown Category")
            for stock in category.get("quotes", []):
                stock_symbol = stock.get("symbol", "N/A")
                price = stock.get("regularMarketPrice", "N/A")

                movers.append({
                    "Category": category_title,
                    "Symbol": stock_symbol,
                    "Price (₹)": price,
                })

        return pd.DataFrame(movers)

    except Exception as e:
        st.error(f"Error fetching market movers: {e}")
        return pd.DataFrame()

# --- Function for Fetching Sector Performance ---
def fetch_sector_performance():
    """Fetch sector performance in the Indian market using yfinance."""
    sectoral_indices = {
        "NIFTY IT": "^CNXIT",
        "NIFTY Pharma": "^CNXPHARMA",
        "NIFTY Bank": "^NSEBANK",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY Auto": "^CNXAUTO",
        "NIFTY Realty": "^CNXREALTY",
        "NIFTY Energy": "^CNXENERGY",
        "NIFTY Infra": "^CNXINFRA",
        "NIFTY Media": "^CNXMEDIA",
        "NIFTY Metal": "^CNXMETAL",
    }

    performance_data = []

    for sector, ticker in sectoral_indices.items():
        try:
            # Fetching last 5 days of data for the index
            data = yf.download(ticker, period="5d", interval="1d")
            
            # Calculating performance as percentage change
            if len(data) > 1:
                initial_price = data['Close'].iloc[0]
                latest_price = data['Close'].iloc[-1]
                performance = ((latest_price - initial_price) / initial_price) * 100
                performance_data.append({
                    "Sector": sector,
                    "Performance (%)": round(performance, 2),
                    "Current Price": round(latest_price, 2)
                })
        except Exception as e:
            st.error(f"Error fetching data for {sector}: {e}")

    # Creating a DataFrame
    df = pd.DataFrame(performance_data)
    if not df.empty:
        df = df.sort_values(by="Performance (%)", ascending=False)  # Sort by performance
    return df

# --- Investment Advice Feature ---
def generate_portfolio_advice(portfolio, total_funds):
    """
    Generate investment advice for the user's portfolio, including buy/hold/sell recommendations 
    and diversification options based on sector performance.
    """
    advice_data = []
    diversification_options = []

    # Fetching sector performance
    sector_performance = fetch_sector_performance()

    for stock in portfolio:
        # Fetching stock data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if data.empty:
            continue

        current_price = data["Close"].iloc[-1]
        invested_amount = stock["invested"]
        qty = stock["qty"]
        stop_loss_price = current_price * (1 - stock["stop_loss_pct"] / 100)

        # Generating Buy/Hold/Sell Advice
        if current_price < stop_loss_price:
            action = "Sell"
            quantity = qty
            reason = f"Price nearing stop-loss of ₹{stop_loss_price:.2f}."
        elif current_price < invested_amount / qty:
            action = "Hold"
            quantity = 0
            reason = "Price is below your average cost; consider holding."
        else:
            action = "Buy"
            quantity = int(0.2 * (total_funds // current_price))  
            reason = "Stock showing growth potential; consider buying more."

        advice_data.append({
            "Stock": stock["symbol"],
            "Current Price (₹)": current_price,
            "Action": action,
            "Quantity": quantity,
            "Reason": reason
        })

    # Generating Diversification Options
    if not sector_performance.empty:
        top_sectors = sector_performance.head(3)  # Top 3 performing sectors
        for _, row in top_sectors.iterrows():
            diversification_options.append({
                "Sector": row["Sector"],
                "Performance (%)": row["Performance (%)"],
                "Suggested Investment (₹)": total_funds * 0.1  # Suggesting 10% of funds for diversification
            })

    return pd.DataFrame(advice_data), pd.DataFrame(diversification_options)


def recommend_profitable_stocks(additional_funds):
    """Recommending profitable stocks based on additional funds available for investment."""
    nifty50_stocks = [
        "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
        "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HDFC.NS", "ITC.NS",
        "MARUTI.NS", "HINDUNILVR.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "WIPRO.NS", "POWERGRID.NS", "ONGC.NS", "NTPC.NS",
        "TATASTEEL.NS", "COALINDIA.NS", "BPCL.NS", "BHARTIARTL.NS", "ADANIENT.NS"
    ]
    recommendations = []

    for ticker in nifty50_stocks:
        try:
            hist = fetch_stock_data(ticker, period="6mo")
            if hist.empty:
                continue

            current_price = hist['Close'].iloc[-1]
            returns = hist['Close'].pct_change().dropna()
            avg_daily_return = returns.mean()
            cumulative_return = ((1 + returns).prod()) - 1

            # Check if the stock fits the additional funds criteria
            if cumulative_return > 0 and current_price <= additional_funds:
                quantity = int(additional_funds // current_price)
                recommendations.append({
                    "Ticker": ticker,
                    "Current Price (₹)": round(current_price, 2),
                    "Cumulative Return (%)": round(cumulative_return * 100, 2),
                    "Average Daily Return (%)": round(avg_daily_return * 100, 2),
                    "Recommended Quantity": quantity
                })
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {e}")

    return pd.DataFrame(recommendations)

# --- Streamlit Layout ---
st.title("PORTFOLIO WIZARD")
st.markdown(
    """
    **Welcome to your personalized stock market advisor!**  
    Use this tool to analyze your portfolio, discover insights, and make informed decisions.
    """
)
# --- Sidebar Inputs ---
st.sidebar.title("Portfolio Inputs")
portfolio = []

# Input for number of stocks in portfolio
num_stocks = st.sidebar.number_input("Number of Stocks in Portfolio", min_value=1, max_value=10, step=1)

# Input: Total funds available for investment (Retained in session state)
if "total_funds" not in st.session_state or st.session_state["total_funds"] < 500.0:
    st.session_state["total_funds"] = 500.0  # Set to minimum allowed value

st.sidebar.markdown("### Total Funds Available for Investment")
st.session_state["total_funds"] = st.sidebar.number_input(
    "Enter total funds available for investment (₹):",
    min_value=500.0,
    step=100.0,
    value=st.session_state["total_funds"],  # Using session state as initial value
    help="This amount will be used to provide investment advice and recommendations."
)

validation_errors = []  # Collect validation errors

# Input for stocks
for i in range(num_stocks):
    st.sidebar.markdown(f"### Stock {i+1}")
    
    # Input fields for each stock
    symbol = st.sidebar.text_input(
        f"Stock Symbol {i+1}",
        key=f"symbol_{i}",
        placeholder="Enter stock symbol (e.g., RELIANCE)"
    ).upper()
    
    qty = st.sidebar.number_input(
        f"Quantity for {symbol or f'Stock {i+1}'}",
        min_value=1,
        key=f"qty_{i}"
    )
    
    invested = st.sidebar.number_input(
        f"Amount Invested in {symbol or f'Stock {i+1}'} (₹)",
        min_value=0.0,
        step=100.0,
        key=f"invested_{i}"
    )
    
    stop_loss_pct = st.sidebar.slider(
        f"Stop Loss % for {symbol or f'Stock {i+1}'}",
        min_value=1,
        max_value=50,
        value=10,
        key=f"stop_loss_{i}"
    )

    # Validation for each stock
    if not symbol:
        validation_errors.append(f"Stock {i+1}: Symbol is missing.")
    if qty <= 0:
        validation_errors.append(f"Stock {i+1}: Quantity must be greater than 0.")
    if invested <= 0:
        validation_errors.append(f"Stock {i+1}: Amount invested must be greater than 0.")
    
    # Adding to portfolio only if all fields are valid
    if symbol and qty > 0 and invested > 0:
        portfolio.append({"symbol": symbol, "qty": qty, "invested": invested, "stop_loss_pct": stop_loss_pct})

# Adding Submit Button
if st.sidebar.button("Submit"):
    # Validation Summary
    if validation_errors:
        st.error("The following errors were found in your inputs:")
        for error in validation_errors:
            st.error(error)
        st.stop()

    # Stoping execution if no valid stocks were added
    if not portfolio:
        st.warning("Add valid stocks to your portfolio to proceed.")
        st.stop()

    # Using session state value for total funds
    total_funds = st.session_state["total_funds"]

    # Initializing variables for calculations
    total_invested = sum(stock["invested"] for stock in portfolio)
    total_current_value = 0
    profit_loss_data = []

    # Processing portfolio data
    for stock in portfolio:
        # Fetch stock data (simulated function here)
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            qty = stock["qty"]
            invested = stock["invested"]
            current_value = qty * current_price
            profit_loss = current_value - invested

            # Appending to profit/loss data
            profit_loss_data.append({
                "Stock": stock["symbol"],
                "Invested (₹)": invested,
                "Current Price (₹)": current_price,
                "Current Value (₹)": current_value,
                "Profit/Loss (₹)": profit_loss,
            })

            total_current_value += current_value

    st.success("Portfolio successfully submitted!")
else:
    st.info("Please fill out the portfolio inputs and press Submit to proceed.")
    st.stop()

# --- Tabs for Features ---
tabs = st.tabs(
    [
        "Portfolio Summary", " Predictions & Technical Indicators", " Sector Allocation",
        " Monte Carlo Simulation", " Trending Sectors & Stocks", " Investment Advice",
        " Profit/Loss Analysis", " Tax Impact", " Risk Management",
        " News Sentiment Analysis", " Final Tip Sheet"
    ]
)

# --- Implementing all  Features ---
# --- Portfolio Summary Tab ---
with tabs[0]:
    st.markdown('<div class="section-header">Portfolio Summary</div>', unsafe_allow_html=True)

    total_invested = sum(stock["invested"] for stock in portfolio) if portfolio else 0
    total_current_value = 0
    cumulative_portfolio_values = []

    # Fetching market index data for comparison
    market_index_data = fetch_stock_data("^NSEI", "6mo")  
    if not market_index_data.empty:
        market_index_data["Cumulative Return"] = (
            (market_index_data["Close"] / market_index_data["Close"].iloc[0]) - 1
        ) * 100

    # Processing portfolio data
    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            qty = stock["qty"]
            invested = stock["invested"]
            current_value = qty * current_price
            cumulative_portfolio_values.append(
                data.assign(Cumulative_Value=qty * data["Close"])
            )
            total_current_value += current_value

    # Creating a cumulative portfolio DataFrame
    if cumulative_portfolio_values:
        portfolio_df = pd.concat(cumulative_portfolio_values).groupby("Date").sum().reset_index()
        portfolio_df["Portfolio Cumulative Return"] = (
            (portfolio_df["Cumulative_Value"] / portfolio_df["Cumulative_Value"].iloc[0]) - 1
        ) * 100

        # Visualizing Portfolio vs. Market Index
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=portfolio_df["Date"],
                y=portfolio_df["Portfolio Cumulative Return"],
                mode="lines",
                name="Portfolio",
            )
        )
        if not market_index_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=market_index_data["Date"],
                    y=market_index_data["Cumulative Return"],
                    mode="lines",
                    name="Market Index",
                )
            )
        fig.update_layout(
            title="Portfolio vs Market Index Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            legend_title="Legend",
        )
        st.plotly_chart(fig)

    # Displaying Portfolio Metrics
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Total Invested (₹)", f"{total_invested:.2f}")
    metrics_cols[1].metric("Total Current Value (₹)", f"{total_current_value:.2f}")
    metrics_cols[2].metric("Overall Profit/Loss (₹)", f"{total_current_value - total_invested:.2f}")
    metrics_cols[3].metric(
        "Portfolio ROI (%)",
        f"{((total_current_value - total_invested) / total_invested) * 100:.2f}" if total_invested > 0 else "N/A"
    )


# --- Feature 2: Predictions & Technical Indicators ---
with tabs[1]:
    st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)

    for stock in portfolio:
        st.subheader(f"Prediction & Technical Indicators for {stock['symbol']}")

        # Fetching historical data
        data = fetch_stock_data(stock["symbol"], "6mo")

        if not data.empty:
            # --- Predictions (Prophet Model) ---
            forecast = predict_prices(data, forecast_days=30)

            # Ensuring predictions start from today and align with trading days
            today = pd.Timestamp.now().normalize()
            trading_days = get_next_trading_days(today, num_days=5)

            # Filtering predictions to only include trading days
            trading_days = pd.to_datetime([str(day.date()) for day in trading_days])
            next_5_days = forecast[forecast["ds"].isin(trading_days)][["ds", "yhat"]]
            next_5_days.columns = ["Date", "Predicted Price (₹)"]

            # Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Historical Data"
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Predicted Prices",
                line=dict(color="blue", dash="dot")
            ))
            fig.update_layout(
                title=f"{stock['symbol']} Historical & Predicted Prices",
                xaxis_title="Date",
                yaxis_title="Price (₹)"
            )
            st.plotly_chart(fig)

            # Displaying next 5 days prediction
            st.write("### Next 5 Days Prediction")
            st.table(next_5_days)

            # --- Technical Indicators ---
            st.subheader("Technical Indicators")

            # Ensuring sufficient data for indicator calculations
            if len(data) < 20:
                st.warning(f"Not enough data to calculate all indicators for {stock['symbol']}. At least 20 rows are required.")
                continue

            # Calculating Indicators
            try:
                # RSI
                data["RSI"] = ta.momentum.rsi(data["Close"], window=14)

                # EMA
                data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)

                # Bollinger Bands
                data["Upper_BB"] = ta.volatility.bollinger_hband(data["Close"])
                data["Lower_BB"] = ta.volatility.bollinger_lband(data["Close"])

                # MACD
                data["MACD"] = ta.trend.macd(data["Close"])
                data["MACD_Signal"] = ta.trend.macd_signal(data["Close"])

                # ADX
                data["ADX"] = ta.trend.adx(data["High"], data["Low"], data["Close"], window=14)

                # ATR
                data["ATR"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=14)

                # Stochastic Oscillator
                data["Stochastic_K"] = ta.momentum.stoch(data["High"], data["Low"], data["Close"], window=14)

                # CCI
                data["CCI"] = ta.trend.cci(data["High"], data["Low"], data["Close"], window=20)

                # Williams %R
                data["Williams_%R"] = ta.momentum.williams_r(data["High"], data["Low"], data["Close"], lbp=14)

                # Custom Volume Oscillator
                data["Volume_Oscillator"] = calculate_volume_oscillator(data)
            except Exception as e:
                st.error(f"Error calculating indicators for {stock['symbol']}: {e}")
                continue

            # Displaying Values
            indicators = {
                "RSI (Relative Strength Index)": data["RSI"].iloc[-1],
                "EMA (20-day Exponential Moving Average)": data["EMA_20"].iloc[-1],
                "Bollinger Bands (Upper)": data["Upper_BB"].iloc[-1],
                "Bollinger Bands (Lower)": data["Lower_BB"].iloc[-1],
                "MACD": data["MACD"].iloc[-1],
                "MACD Signal": data["MACD_Signal"].iloc[-1],
                "ADX (Average Directional Index)": data["ADX"].iloc[-1],
                "ATR (Average True Range)": data["ATR"].iloc[-1],
                "Stochastic Oscillator %K": data["Stochastic_K"].iloc[-1],
                "CCI (Commodity Channel Index)": data["CCI"].iloc[-1],
                "Williams %R": data["Williams_%R"].iloc[-1],
                "Volume Oscillator": data["Volume_Oscillator"].iloc[-1],
            }
            indicator_df = pd.DataFrame(indicators.items(), columns=["Indicator", "Value"])
            st.table(indicator_df)

            # --- Explanations & Portfolio Impact ---
            st.subheader("How These Indicators Impact Your Portfolio")

            insights = []

            # RSI Insights
            rsi = indicators["RSI (Relative Strength Index)"]
            if rsi > 70:
                insights.append(f"RSI is {rsi:.2f}, indicating {stock['symbol']} may be overbought. "
                                 f"Consider reducing exposure.")
            elif rsi < 30:
                insights.append(f"RSI is {rsi:.2f}, indicating {stock['symbol']} may be oversold. "
                                 f"Consider buying or holding.")

            # Bollinger Bands Insights
            upper_bb = indicators["Bollinger Bands (Upper)"]
            lower_bb = indicators["Bollinger Bands (Lower)"]
            if current_price >= upper_bb:
                insights.append(f"{stock['symbol']} is trading near the upper Bollinger Band ({upper_bb:.2f}), "
                                 f"indicating overbought conditions. This could mean reduced future growth; consider selling.")
            elif current_price <= lower_bb:
                insights.append(f"{stock['symbol']} is trading near the lower Bollinger Band ({lower_bb:.2f}), "
                                 f"suggesting oversold conditions. This could signal a buying opportunity.")

            # MACD Insights
            macd = indicators["MACD"]
            macd_signal = indicators["MACD Signal"]
            if macd > macd_signal:
                insights.append(f"The MACD ({macd:.2f}) is above the Signal Line ({macd_signal:.2f}), "
                                 f"indicating bullish momentum for {stock['symbol']}. Consider holding or buying.")
            else:
                insights.append(f"The MACD ({macd:.2f}) is below the Signal Line ({macd_signal:.2f}), "
                                 f"suggesting bearish momentum. Reassess your position in {stock['symbol']}.")

            # ADX Insights
            adx = indicators["ADX (Average Directional Index)"]
            if adx > 25:
                insights.append(f"ADX is {adx:.2f}, indicating a strong trend in {stock['symbol']}. "
                                 f"This trend can support your investment strategy.")
            elif adx < 20:
                insights.append(f"ADX is {adx:.2f}, indicating a weak trend in {stock['symbol']}. "
                                 f"Be cautious as the stock might lack clear direction.")

            # ATR Insights
            atr = indicators["ATR (Average True Range)"]
            if atr > (0.02 * data["Close"].iloc[-1]):
                insights.append(f"ATR is {atr:.2f}, indicating high volatility in {stock['symbol']}. "
                                 f"Consider managing risk by adjusting your position size.")
            else:
                insights.append(f"ATR is {atr:.2f}, indicating low volatility in {stock['symbol']}. "
                                 f"Risk levels are stable.")

            # Stochastic Oscillator Insights
            stoch_k = indicators["Stochastic Oscillator %K"]
            if stoch_k > 80:
                insights.append(f"Stochastic Oscillator %K is {stoch_k:.2f}, suggesting overbought conditions for {stock['symbol']}. "
                                 f"Consider taking profits.")
            elif stoch_k < 20:
                insights.append(f"Stochastic Oscillator %K is {stoch_k:.2f}, suggesting oversold conditions for {stock['symbol']}. "
                                 f"Consider buying opportunities.")

            # CCI Insights
            cci = indicators["CCI (Commodity Channel Index)"]
            if cci > 100:
                insights.append(f"CCI is {cci:.2f}, indicating overbought conditions for {stock['symbol']}. "
                                 f"Consider reducing exposure.")
            elif cci < -100:
                insights.append(f"CCI is {cci:.2f}, indicating oversold conditions for {stock['symbol']}. "
                                 f"Consider adding to your position.")

            # Displaying Insights
            for i, insight in enumerate(insights, 1):
                st.write(f"**{i}. {insight}**")

            # --- Actionable Suggestions ---
            st.subheader("Actions to Take")
            st.write("""
                Based on the current technical indicators, consider the following actions:
                - **Monitor RSI**: If it's nearing overbought (70) or oversold (30) conditions, adjust your position accordingly.
                - **Watch Price Trends**: Compare current prices with SMA and Bollinger Bands to identify potential reversals or breakouts.
                - **Track Momentum**: Use MACD crossovers to confirm bullish or bearish trends.
                - **Analyze Volatility**: Use ATR and ADX to understand risk and trend strength.
            """)

# --- Feature 3: Sector Allocation ---
with tabs[2]:
    st.markdown('<div class="section-header">Sector Allocation</div>', unsafe_allow_html=True)

    sector_data = {}
    for stock in portfolio:
        try:
            stock_info = yf.Ticker(stock["symbol"]).info
            sector = stock_info.get("sector", "Unknown")
            if sector != "Unknown":
                sector_data[sector] = sector_data.get(sector, 0) + stock["invested"]
        except Exception as e:
            st.error(f"Error fetching sector data for {stock['symbol']}: {e}")

    if sector_data:
        sector_df = pd.DataFrame(list(sector_data.items()), columns=["Sector", "Investment (₹)"])
        fig = px.pie(sector_df, values="Investment (₹)", names="Sector", title="Sector Allocation")
        st.plotly_chart(fig)

        st.markdown("""
            **What this means:**  
            - This chart shows how your investments are distributed across sectors.  
            - Use it to identify over-concentration or under-diversification.
        """)
    else:
        st.write("No sector data available for your portfolio.")

# --- Feature 4: Monte Carlo Simulation ---
with tabs[3]:
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)

    for stock in portfolio:
        st.subheader(f"Monte Carlo Simulation for {stock['symbol']}")

        # Fetching historical data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            # Running Monte Carlo Simulation
            simulations = monte_carlo_simulation(data, days=30, simulations=1000)

            # Calculating Statistical Metrics
            mean_price = np.mean(simulations)
            median_price = np.median(simulations)
            lower_ci = np.percentile(simulations, 2.5)  # 2.5th percentile for 95% confidence interval
            upper_ci = np.percentile(simulations, 97.5)  # 97.5th percentile for 95% confidence interval

            # Plotting Histogram with Annotations
            fig = px.histogram(
                simulations,
                nbins=50,
                title=f"Monte Carlo Simulation for {stock['symbol']}",
                labels={"value": "Simulated Price (₹)", "count": "Frequency"}
            )
            fig.add_vline(x=mean_price, line_width=2, line_dash="dash", line_color="blue", annotation_text="Mean")
            fig.add_vline(x=median_price, line_width=2, line_dash="dash", line_color="green", annotation_text="Median")
            fig.add_vline(x=lower_ci, line_width=2, line_dash="dot", line_color="red", annotation_text="2.5% CI")
            fig.add_vline(x=upper_ci, line_width=2, line_dash="dot", line_color="red", annotation_text="97.5% CI")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

            # Displaying Metrics
            st.write("### Statistical Metrics")
            st.write(f"**Mean (Average Price):** ₹{mean_price:.2f}")
            st.write(f"**Median (Middle Price):** ₹{median_price:.2f}")
            st.write(f"**95% Confidence Interval:** ₹{lower_ci:.2f} to ₹{upper_ci:.2f}")

            # Textual Explanation
            st.write("### What This Means")
            st.markdown(f"""
                - The histogram shows the distribution of possible prices for {stock['symbol']} after 30 days based on historical trends.
                - The **mean price** (blue line) is the average of all simulated outcomes.
                - The **median price** (green line) represents the middle simulated value, indicating that 50% of the simulations are below this price and 50% are above.
                - The **95% confidence interval** (red lines) indicates that there is a 95% chance the price will fall between ₹{lower_ci:.2f} and ₹{upper_ci:.2f}.
            """)

            # Visual Explanation
            st.write("### Visual Explanation")
            st.markdown("""
                - The histogram's shape shows the range and frequency of potential prices.
                - Use the confidence interval to assess the potential downside and upside risk of the stock.
            """)
            st.write("#### Suggested Actions")
            st.markdown(f"""
                - If the lower confidence interval is significantly below your stop-loss level, you may consider reducing exposure.
                - If the mean or median price aligns with your profit goals, consider holding your position.
                - Review other technical indicators for confirmation.
            """)

# --- Feature 5: Trending Stocks and Sectors ---
with tabs[4]:
    st.markdown('<div class="section-header">Trending Stocks</div>', unsafe_allow_html=True)

    # --- Market Movers Section ---
    st.subheader(" Market Movers (Trending Stocks)")

    # Fetching market movers
    market_movers_df = fetch_market_movers()

    if not market_movers_df.empty:
        # Grouping by category
        for category in market_movers_df["Category"].unique():
            st.subheader(category)  # Display category (e.g., Top Gainers, Top Losers)
            category_data = market_movers_df[market_movers_df["Category"] == category]

            # Creating cards for each stock
            cols = st.columns(3)  # Displaying cards in a grid (3 per row)
            for index, row in category_data.iterrows():
                col = cols[index % 3]  # Distributing cards across columns
                with col:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                            <h3 style="margin: 0; color: #333;">{row['Symbol']}</h3>
                            <p style="font-size: 20px; color: #4caf50; font-weight: bold;">₹{row['Price (₹)']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.warning("No market movers data available. Try again later.")

    # --- Trending Sectors Section ---
    st.subheader("Trending Sectors")
    try:
        # Fetching sector performance
        sectors_df = fetch_sector_performance()

        if not sectors_df.empty:
            st.write("### Sector Performance (Last 5 Days)")
            st.dataframe(sectors_df.style.format({"Performance (%)": "{:.2f}%", "Current Price": "₹{:.2f}"}))

            # Creating Grouped Bar Chart
            fig = px.bar(
                sectors_df,
                x="Sector",
                y="Performance (%)",
                color="Sector",
                text="Performance (%)",
                title="Sectoral Performance (Grouped Bar Chart)",
                labels={"Performance (%)": "Change (%)", "Sector": "Sector Name"}
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig)
        else:
            st.warning("No sector performance data available. Try again later.")
    except Exception as e:
        st.error(f"Error fetching sector performance: {e}")


# --- Investment Advice Tab ---
with tabs[5]:  # Ensure this is part of the Investment Advice tab
    st.markdown('<div class="section-header">Investment Advice</div>', unsafe_allow_html=True)

    # Generating portfolio advice and diversification options
    portfolio_advice, diversification_options = generate_portfolio_advice(portfolio, st.session_state["total_funds"])

    # Displaying portfolio-specific advice
    st.subheader("Portfolio Recommendations (Buy/Hold/Sell)")
    if not portfolio_advice.empty:
        # Display in grid format using st.metric
        st.write("### Recommendations for Your Portfolio")
        cols = st.columns(3)  # Creating 3 columns for a grid layout
        for index, row in portfolio_advice.iterrows():
            col = cols[index % 3]  # Cycle through columns
            with col:
                st.metric(
                    label=f"**{row['Stock']}**",
                    value=row['Action'],
                    delta=f"{row['Quantity']} shares",
                    help=row['Reason']
                )
    else:
        st.warning("No advice available for your portfolio.")

    # Visualizing  Buy/Hold/Sell Recommendations
    st.write("### Buy/Hold/Sell Recommendations Summary")
    if not portfolio_advice.empty:
        # Group by action
        action_counts = portfolio_advice.groupby("Action").sum(numeric_only=True).reset_index()

        # Creating a bar chart for Buy/Hold/Sell quantities
        fig = px.bar(
            action_counts,
            x="Action",
            y="Quantity",
            title="Buy/Hold/Sell Recommendations",
            text="Quantity",
            labels={"Action": "Recommendation", "Quantity": "Number of Shares"},
            color="Action"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)

    # Displaying diversification options
    st.subheader("Diversification Opportunities")
    if not diversification_options.empty:
        st.write("### Sector Diversification Suggestions")
        for index, row in diversification_options.iterrows():
            st.metric(
                label=f"{row['Sector']}",
                value=f"{row['Performance (%)']}%",
                delta=f"₹{row['Suggested Investment (₹)']}",
                help="Based on recent performance."
            )
    else:
        st.warning("No diversification options available.")

    # Showing recommendations for additional investment
    st.subheader("Recommended Stocks for Additional Investment")
    if st.session_state["total_funds"] > 0:
        recommendations_df = recommend_profitable_stocks(st.session_state["total_funds"])
        if not recommendations_df.empty:
            st.write("### Suggested Stocks for Investment")
            cols = st.columns(3)  # Displaying in a grid format
            for index, row in recommendations_df.iterrows():
                col = cols[index % 3]  # Cycling through columns
                with col:
                    st.metric(
                        label=f"**{row['Ticker']}**",
                        value=f"₹{row['Current Price (₹)']}",
                        delta=f"{row['Recommended Quantity']} shares",
                        help=f"Cumulative Return: {row['Cumulative Return (%)']}%"
                    )
        else:
            st.warning("No stocks available for recommendation.")
    else:
        st.warning("Please enter a valid total amount to see recommendations.")




# --- Updated Profit/Loss Analysis ---
with tabs[6]:
    st.markdown('<div class="section-header">Profit/Loss Analysis</div>', unsafe_allow_html=True)

    total_invested = sum(stock["invested"] for stock in portfolio)
    total_current_value = 0
    profit_loss_data = []

    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            invested = stock["invested"]
            qty = stock["qty"]
            current_value = qty * current_price
            profit_loss = current_value - invested

            # Calculating additional metrics
            percentage_change = (profit_loss / invested) * 100 if invested != 0 else 0
            break_even_price = invested / qty if qty != 0 else 0
            holding_period = 6  # Placeholder: replace with actual holding period
            annualized_return = (
                ((current_value / invested) ** (1 / (holding_period / 12))) - 1
                if invested > 0
                else 0
            )

            total_current_value += current_value
            profit_loss_data.append({
                "Stock": stock["symbol"],
                "Invested (₹)": invested,
                "Current Price (₹)": current_price,
                "Current Value (₹)": current_value,
                "Profit/Loss (₹)": profit_loss,
                "Percentage Change (%)": round(percentage_change, 2),
                "Break-Even Price (₹)": break_even_price,
                "Annualized Return (%)": round(annualized_return * 100, 2)
            })

    profit_loss_df = pd.DataFrame(profit_loss_data)
    st.dataframe(profit_loss_df)

    # Displaying Metrics
    st.metric("Total Invested (₹)", total_invested)
    st.metric("Total Current Value (₹)", total_current_value)
    st.metric("Overall Profit/Loss (₹)", total_current_value - total_invested)
    st.metric("Portfolio ROI (%)", round(((total_current_value - total_invested) / total_invested) * 100, 2))

    # Pie Chart: Profit/Loss Distribution
    st.subheader("Profit/Loss Distribution")
    profit_loss_pie = profit_loss_df[["Stock", "Profit/Loss (₹)"]]
    fig_pie = px.pie(
        profit_loss_pie,
        values="Profit/Loss (₹)",
        names="Stock",
        title="Profit/Loss Distribution Across Portfolio",
        hole=0.4,
    )
    st.plotly_chart(fig_pie)

    # Trend Line: Cumulative Portfolio Value Over Time
    st.subheader("Cumulative Portfolio Value Over Time")
    cumulative_values = []
    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            data["Cumulative Value"] = stock["qty"] * data["Close"]
            cumulative_values.append(data[["Date", "Cumulative Value"]])

    if cumulative_values:
        cumulative_df = pd.concat(cumulative_values).groupby("Date").sum().reset_index()
        fig_line = px.line(
            cumulative_df,
            x="Date",
            y="Cumulative Value",
            title="Portfolio Cumulative Value Over Time",
            labels={"Cumulative Value": "Portfolio Value (₹)"},
        )
        st.plotly_chart(fig_line)

# --- Tax Impact Analysis ---
with tabs[7]:
    st.markdown('<div class="section-header">Tax Impact</div>', unsafe_allow_html=True)

    tax_data = []
    total_stcg_tax = 0
    total_ltcg_tax = 0
    total_dividend_tax = 0
    total_tax = 0

    for stock in portfolio:
        # Fetch stock data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            invested = stock["invested"]
            qty = stock["qty"]
            profit = (current_price * qty) - invested
            holding_period = 6  
            dividend_income = stock.get("dividend", 0)  # dividend income per stock

            if profit > 0:
                if holding_period < 12:
                    # Short-Term Capital Gains Tax
                    stcg_tax = profit * 0.15
                    total_stcg_tax += stcg_tax
                    tax_data.append({
                        "Stock": stock["symbol"],
                        "Profit (₹)": profit,
                        "Category": "Short-Term (STCG)",
                        "Tax (₹)": stcg_tax,
                        "Holding Period (Months)": holding_period
                    })
                else:
                    # Long-Term Capital Gains Tax
                    taxable_gain = max(0, profit - 100000)  # Exemption limit for LTCG
                    ltcg_tax = taxable_gain * 0.10
                    total_ltcg_tax += ltcg_tax
                    tax_data.append({
                        "Stock": stock["symbol"],
                        "Profit (₹)": profit,
                        "Category": "Long-Term (LTCG)",
                        "Tax (₹)": ltcg_tax,
                        "Holding Period (Months)": holding_period
                    })

            else:
                # Losses
                tax_data.append({
                    "Stock": stock["symbol"],
                    "Profit (₹)": profit,
                    "Category": "Loss",
                    "Tax (₹)": 0,
                    "Holding Period (Months)": holding_period
                })

            # Dividend Taxation
            if dividend_income > 0:
                dividend_tax = dividend_income * 0.30  # Assume 30% slab rate for simplicity
                total_dividend_tax += dividend_tax
                tax_data.append({
                    "Stock": stock["symbol"],
                    "Profit (₹)": 0,
                    "Category": "Dividend",
                    "Tax (₹)": dividend_tax,
                    "Holding Period (Months)": "-",
                })

    # Calculating Total Tax
    total_tax = total_stcg_tax + total_ltcg_tax + total_dividend_tax
    total_tax_with_cess = total_tax * 1.04  # Adding 4% cess

    # Displaying Tax Breakdown
    tax_df = pd.DataFrame(tax_data)
    st.subheader("Detailed Tax Breakdown")
    st.dataframe(tax_df)

    # Displaying Metrics
    st.metric("Total Short-Term Capital Gains Tax (₹)", round(total_stcg_tax, 2))
    st.metric("Total Long-Term Capital Gains Tax (₹)", round(total_ltcg_tax, 2))
    st.metric("Total Dividend Tax (₹)", round(total_dividend_tax, 2))
    st.metric("Total Tax with Cess (₹)", round(total_tax_with_cess, 2))

    # Visualizing Tax Distribution
    st.subheader("Tax Distribution")
    tax_distribution = {
        "Short-Term Gains Tax (₹)": total_stcg_tax,
        "Long-Term Gains Tax (₹)": total_ltcg_tax,
        "Dividend Tax (₹)": total_dividend_tax,
    }
    fig_pie_tax = px.pie(
        values=tax_distribution.values(),
        names=tax_distribution.keys(),
        title="Tax Distribution by Category",
        hole=0.4,
    )
    st.plotly_chart(fig_pie_tax)

    # Insights and Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
        - **Optimize Taxation**: Use losses to offset short-term or long-term gains where applicable.
        - **LTCG Exemption**: Utilize the ₹1,00,000 exemption limit effectively.
        - **Plan Dividend Income**: Keep track of dividends to manage overall tax liabilities.
        - **Carry-Forward Losses**: If you have losses, ensure to declare them to carry forward for up to 8 years.
        - **Consult a Tax Advisor**: For complex portfolios, seek professional advice to minimize tax liabilities.
    """)

# --- Risk Management Analysis ---
with tabs[8]:
    st.markdown('<div class="section-header">Risk Management</div>', unsafe_allow_html=True)

    # Portfolio Risk Metrics
    confidence_level = 0.95
    risk_free_rate = RISK_FREE_RATE
    portfolio_var = 0
    portfolio_cvar = 0
    portfolio_beta = 0
    portfolio_volatility = 0
    individual_risks = []
    stock_betas = []
    returns_data = []

    for stock in portfolio:
        # Fetching historical data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            returns = data["Close"].pct_change().dropna()
            returns_data.append(returns)

            # Calculating VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            invested = stock["invested"]
            stock_var = invested * var
            portfolio_var += stock_var

            # Calculating CVaR
            cvar = returns[returns <= var].mean()
            stock_cvar = invested * cvar
            portfolio_cvar += stock_cvar

            # Simulated Beta (Placeholder)
            beta = 1.2  
            stock_betas.append(beta)
            portfolio_beta += beta * (invested / sum(stock["invested"] for stock in portfolio))

            # Annualized Volatility
            stock_volatility = returns.std() * np.sqrt(252)
            portfolio_volatility += (invested / sum(stock["invested"] for stock in portfolio)) * stock_volatility

            individual_risks.append({
                "Stock": stock["symbol"],
                "VaR (₹)": round(stock_var, 2),
                "CVaR (₹)": round(stock_cvar, 2),
                "Beta": beta,
                "Annualized Volatility (%)": round(stock_volatility * 100, 2),
            })

    # Aggregated Risk Metrics
    risk_df = pd.DataFrame(individual_risks)
    st.subheader("Individual Stock Risk Metrics")
    st.dataframe(risk_df)

    # Portfolio-Level Metrics
    sharpe_ratio = (
        (total_current_value - total_invested) / total_invested - risk_free_rate
    ) / portfolio_volatility if portfolio_volatility != 0 else 0

    st.subheader("Portfolio-Level Risk Metrics")
    st.metric("Portfolio VaR (₹)", round(portfolio_var, 2))
    st.metric("Portfolio CVaR (₹)", round(portfolio_cvar, 2))
    st.metric("Portfolio Beta", round(portfolio_beta, 2))
    st.metric("Portfolio Volatility (%)", round(portfolio_volatility * 100, 2))
    st.metric("Portfolio Sharpe Ratio", round(sharpe_ratio, 2))

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    if returns_data:
        correlation_matrix = pd.concat(returns_data, axis=1).corr()
        correlation_matrix.columns = [stock["symbol"] for stock in portfolio]
        correlation_matrix.index = correlation_matrix.columns

        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Portfolio Correlation Matrix",
            labels=dict(color="Correlation"),
        )
        st.plotly_chart(fig_corr)

    # Stress Testing
    st.subheader("Stress Testing: Market Downturn")
    downturn_scenario = -0.2  # Simulating a 20% market drop
    stress_impact = []
    for stock in portfolio:
        stress_loss = stock["invested"] * downturn_scenario
        stress_impact.append({"Stock": stock["symbol"], "Stress Loss (₹)": round(stress_loss, 2)})

    stress_df = pd.DataFrame(stress_impact)
    st.dataframe(stress_df)

    # Visualizations
    st.subheader("Risk Visualization")

    # Bar Chart for VaR
    fig_var = px.bar(
        risk_df,
        x="Stock",
        y="VaR (₹)",
        title="Value at Risk (VaR) by Stock",
        labels={"VaR (₹)": "Value at Risk (₹)", "Stock": "Stock Name"},
    )
    st.plotly_chart(fig_var)

    # Portfolio Stress Line Chart
    cumulative_stress_loss = stress_df["Stress Loss (₹)"].sum()
    stress_fig = px.line(
        x=["Current Portfolio Value", "After Stress Test"],
        y=[total_current_value, total_current_value + cumulative_stress_loss],
        title="Portfolio Value Under Stress Test",
        labels={"x": "Scenario", "y": "Portfolio Value (₹)"},
    )
    st.plotly_chart(stress_fig)

    # Insights and Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
        - **Diversify**: Minimize correlated assets to reduce concentrated risks.
        - **Monitor Beta**: A high portfolio beta indicates sensitivity to market movements.
        - **Stress Test Regularly**: Prepare for market downturns by analyzing potential impacts.
        - **Enhance Sharpe Ratio**: Aim for higher risk-adjusted returns by managing volatility.
    """)


    # --- Enhanced News Sentiment Analysis ---
with tabs[9]:
    st.markdown('<div class="section-header">News Sentiment Analysis & Recent News</div>', unsafe_allow_html=True)

    #  User-Specific News Sentiments
    st.subheader("News Sentiments Related to Your Portfolio")
    sentiment_data = []
    sentiment_trends = []

    for stock in portfolio:
        # Fetching news articles for the stock
        news_articles = fetch_google_news(stock["symbol"])
        sentiment_scores = []

        for article in news_articles:
            # Performing sentiment analysis
            text_blob_analysis = TextBlob(article["title"])
            polarity = text_blob_analysis.sentiment.polarity
            subjectivity = text_blob_analysis.sentiment.subjectivity

            # Enhancing sentiment analysis using VADER
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            vader_scores = analyzer.polarity_scores(article["title"])
            compound_score = vader_scores['compound']

            # Weighted Sentiment Score (average of TextBlob and VADER)
            weighted_score = (polarity + compound_score) / 2
            sentiment_scores.append(weighted_score)

            sentiment_trends.append({
                "Stock": stock["symbol"],
                "Date": article["published"],
                "Title": article["title"],
                "Polarity": polarity,
                "Subjectivity": subjectivity,
                "Compound (VADER)": compound_score,
                "Weighted Sentiment": weighted_score
            })

        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_label = (
                "Strongly Positive" if avg_sentiment > 0.5 else
                "Positive" if avg_sentiment > 0.1 else
                "Neutral" if avg_sentiment >= -0.1 else
                "Negative" if avg_sentiment >= -0.5 else
                "Strongly Negative"
            )
        else:
            avg_sentiment = 0
            sentiment_label = "Neutral"

        sentiment_data.append({
            "Stock": stock["symbol"],
            "Average Sentiment Score": round(avg_sentiment, 2),
            "Sentiment Label": sentiment_label,
            "Recent Articles": [article["title"] for article in news_articles]
        })

    # Displaying Portfolio Sentiments
    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df)

    # Visualizing Portfolio Sentiment Distribution
    st.subheader("Portfolio Sentiment Distribution")
    fig_sentiment_pie = px.pie(
        sentiment_df,
        values="Average Sentiment Score",
        names="Stock",
        title="Sentiment Distribution for Your Portfolio",
        hole=0.4
    )
    st.plotly_chart(fig_sentiment_pie)

    # Sentiment Trends Over Time
    sentiment_trends_df = pd.DataFrame(sentiment_trends)
    st.subheader("Sentiment Trends for Your Portfolio")
    fig_sentiment_line = px.line(
        sentiment_trends_df,
        x="Date",
        y="Weighted Sentiment",
        color="Stock",
        title="Sentiment Trends Over Time",
        labels={"Weighted Sentiment": "Sentiment Score", "Date": "Publication Date"}
    )
    st.plotly_chart(fig_sentiment_line)

    #  Recent General News
    st.subheader("Recent News on Indian Stock Market")
    def fetch_recent_market_news():
        rss_url = "https://news.google.com/rss/search?q=Indian+Stock+Market&hl=en-IN&gl=IN&ceid=IN:en"
        news_feed = feedparser.parse(rss_url)
        articles = []
        for entry in news_feed.entries[:10]:  # Fetch top 10 articles
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })
        return articles

    recent_news = fetch_recent_market_news()

    if recent_news:
        for article in recent_news:
            st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                    <h4 style="margin: 0; color: #007bff;">{article['title']}</h4>
                    <p style="margin: 5px 0; color: #555;">Published: {article['published']}</p>
                    <a href="{article['link']}" target="_blank" style="color: #007bff; text-decoration: none;">Read Full Article</a>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No recent news available. Please try again later.")
# --- Final Tip Sheet ---
with tabs[10]:
    st.header("Final Tip Sheet")

    # Portfolio Summary
    st.subheader("Portfolio Summary")
    st.metric("Total Invested (₹)", total_invested)
    st.metric("Total Current Value (₹)", total_current_value)
    st.metric("Overall Profit/Loss (₹)", total_current_value - total_invested)
    st.metric("Portfolio ROI (%)", round(((total_current_value - total_invested) / total_invested) * 100, 2))

    # Sector Performance Visualization
    st.subheader("Sector Performance")
    if 'sectors_df' in locals() and not sectors_df.empty:
        st.write("### Top Performing Sectors in Your Portfolio")
        # Sunburst Chart for Sector Performance
        fig_sunburst = px.sunburst(
            sectors_df,
            path=["Sector"],
            values="Performance (%)",
            title="Sector Performance Distribution",
            color="Performance (%)",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_sunburst)
    else:
        st.warning("Sector performance data unavailable.")

    # Investment Recommendations Visualization
    st.subheader("Investment Recommendations")
    if not portfolio_advice.empty:
        st.write("### Recommended Actions for Your Portfolio")
        # Pie Chart for Buy/Hold/Sell
        action_counts = portfolio_advice.groupby("Action").sum(numeric_only=True).reset_index()
        fig_recommendations_pie = px.pie(
            action_counts,
            values="Quantity",
            names="Action",
            title="Buy/Hold/Sell Recommendations",
            hole=0.4,
            color="Action",
            color_discrete_map={"Buy": "green", "Hold": "blue", "Sell": "red"}
        )
        st.plotly_chart(fig_recommendations_pie)
    else:
        st.warning("No investment advice available.")

    # Risk Assessment Visualization
    st.subheader("Risk Assessment")
    if not risk_df.empty:
        st.write("### Individual Stock Risk Metrics")
        # Radar Chart for Risk Metrics
        fig_radar = px.line_polar(
            risk_df,
            r="VaR (₹)",
            theta="Stock",
            line_close=True,
            title="Value at Risk (VaR) Across Stocks",
            color="Stock"
        )
        st.plotly_chart(fig_radar)

        # Risk-Level Bar Chart
        fig_risk_bar = px.bar(
            risk_df,
            x="Stock",
            y=["VaR (₹)", "CVaR (₹)"],
            barmode="group",
            title="Risk Metrics Comparison (VaR vs CVaR)",
            labels={"value": "Risk Amount (₹)", "variable": "Risk Metric"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_risk_bar)

        st.write("### Portfolio-Level Risk Metrics")
        st.metric("Portfolio VaR (₹)", round(portfolio_var, 2))
        st.metric("Portfolio CVaR (₹)", round(portfolio_cvar, 2))
        st.metric("Portfolio Beta", round(portfolio_beta, 2))
        st.metric("Portfolio Volatility (%)", round(portfolio_volatility * 100, 2))
        st.metric("Portfolio Sharpe Ratio", round(sharpe_ratio, 2))

        st.markdown("""
        **Insights**:
        - **VaR (Value at Risk):** Indicates the potential loss in a worst-case scenario over a given time frame.
        - **CVaR (Conditional VaR):** Average loss in extreme scenarios (tail risk).
        - **Beta:** Measures sensitivity to market movements. Values > 1 mean higher sensitivity.
        - **Sharpe Ratio:** Higher values indicate better risk-adjusted returns.
        """)
    else:
        st.warning("Risk metrics unavailable.")

    # News Sentiment Highlights
    st.subheader("News Sentiment Highlights")
    for index, row in sentiment_df.iterrows():
        st.write(f"**{row['Stock']}:** {row['Sentiment Label']} sentiment. "
                 f"Top Articles: {', '.join(row['Recent Articles'][:3])}")

    # Tax Impact
    st.subheader("Tax Impact")
    st.metric("Short-Term Capital Gains Tax (₹)", round(total_stcg_tax, 2))
    st.metric("Long-Term Capital Gains Tax (₹)", round(total_ltcg_tax, 2))
    st.metric("Dividend Tax (₹)", round(total_dividend_tax, 2))
    st.metric("Total Tax Liability with Cess (₹)", round(total_tax_with_cess, 2))

    st.markdown("""
    **Tax Tips**:
    - Use losses to offset gains where applicable.
    - Ensure to leverage the ₹1,00,000 LTCG exemption.
    - Declare losses to carry forward for up to 8 years.
    """)

    # Stress Test Results
    st.subheader("Stress Test Results")
    if "stress_df" in locals():
        fig_stress_line = px.line(
            x=["Current Portfolio Value", "After Stress Test"],
            y=[total_current_value, total_current_value + cumulative_stress_loss],
            title="Portfolio Value Under Stress Test",
            labels={"x": "Scenario", "y": "Portfolio Value (₹)"}
        )
        st.plotly_chart(fig_stress_line)
        st.write(f"Total Portfolio Value after a 20% market downturn: ₹{total_current_value + cumulative_stress_loss:.2f}")

    # General Tips
    st.subheader("General Tips")
    st.write("- Monitor technical indicators and sentiment trends regularly.")
    st.write("- Diversify investments across sectors to reduce risk.")
    st.write("- Use Monte Carlo simulations and risk metrics for better planning.")
    st.write("- Consult a tax advisor for efficient tax management.")
