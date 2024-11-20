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
    """Predict stock prices using Prophet."""
    df = data.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast

def fetch_google_news(stock_symbol):
    """Fetch news articles related to a stock symbol using Google RSS feed."""
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
    """Optimize portfolio weights using the Sharpe Ratio."""
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
    """Fetch non-trading days by checking gaps in historical stock data."""
    try:
        # Download historical data for the NIFTY 50 index
        historical_data = yf.download("^NSEI", period="1y", interval="1d")
        
        # Extract all dates from the period and check for missing dates
        all_dates = pd.date_range(start=historical_data.index.min(), end=historical_data.index.max())
        trading_days = historical_data.index  # Actual trading days
        non_trading_days = set(all_dates) - set(trading_days)
        
        # Return non-trading days as strings
        return set(non_trading_day.strftime('%Y-%m-%d') for non_trading_day in non_trading_days)
    except Exception as e:
        st.error(f"Error fetching market holidays: {e}")
        return set()  # Return an empty set on failure

def get_next_trading_days(start_date, num_days=5):
    """Fetch the next 'num_days' trading days starting from a given date."""
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
# --- Fetch Market Movers ---
def fetch_market_movers(region="IN", lang="en-US", start=0, count=10):
    """Fetch market movers (top gainers, losers, active stocks) using Yahoo Finance API."""
    conn = http.client.HTTPSConnection("apidojo-yahoo-finance-v1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5d63bb22bemshb6e582f5cdfd2cdp1d4344jsn14c1f5b16633",
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    try:
        # Request market movers
        conn.request("GET", f"/market/v2/get-movers?region={region}&lang={lang}&start={start}&count={count}", headers=headers)
        res = conn.getresponse()
        data = res.read()

        # Decode and parse the response
        response_json = json.loads(data.decode("utf-8"))

        # Extract movers data
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

# --- Function to Fetch Sector Performance ---
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
            # Fetch the last 5 days of data for the index
            data = yf.download(ticker, period="5d", interval="1d")
            
            # Calculate performance as percentage change
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

    # Create a DataFrame
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

    # Fetch sector performance
    sector_performance = fetch_sector_performance()

    for stock in portfolio:
        # Fetch stock data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if data.empty:
            continue

        current_price = data["Close"].iloc[-1]
        invested_amount = stock["invested"]
        qty = stock["qty"]
        stop_loss_price = current_price * (1 - stock["stop_loss_pct"] / 100)

        # Generate Buy/Hold/Sell Advice
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
            quantity = int(0.2 * (total_funds // current_price))  # Example allocation
            reason = "Stock showing growth potential; consider buying more."

        advice_data.append({
            "Stock": stock["symbol"],
            "Current Price (₹)": current_price,
            "Action": action,
            "Quantity": quantity,
            "Reason": reason
        })

    # Generate Diversification Options
    if not sector_performance.empty:
        top_sectors = sector_performance.head(3)  # Top 3 performing sectors
        for _, row in top_sectors.iterrows():
            diversification_options.append({
                "Sector": row["Sector"],
                "Performance (%)": row["Performance (%)"],
                "Suggested Investment (₹)": total_funds * 0.1  # Suggest 10% of funds for diversification
            })

    return pd.DataFrame(advice_data), pd.DataFrame(diversification_options)


def recommend_profitable_stocks(additional_funds):
    """Recommend profitable stocks based on additional funds available for investment."""
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
st.set_page_config(page_title="Smart Portfolio Advisor", layout="wide")
st.title("Smart Portfolio Advisor 📈")
st.markdown(
    """
    **Welcome to your personalized stock market advisor!**  
    Use this tool to analyze your portfolio, discover insights, and make informed decisions.
    """
)

# --- Sidebar Inputs ---
st.sidebar.title("Portfolio Inputs")
portfolio = []
num_stocks = st.sidebar.number_input("Number of Stocks in Portfolio", min_value=1, max_value=10, step=1)

for i in range(num_stocks):
    st.sidebar.markdown(f"### Stock {i+1}")
    symbol = st.sidebar.text_input(f"Stock Symbol {i+1}", key=f"symbol_{i}").upper()
    qty = st.sidebar.number_input(f"Quantity for {symbol}", min_value=1, key=f"qty_{i}")
    invested = st.sidebar.number_input(f"Amount Invested in {symbol} (₹)", min_value=0.0, step=100.0, key=f"invested_{i}")
    stop_loss_pct = st.sidebar.slider(f"Stop Loss % for {symbol}", min_value=1, max_value=50, value=10, key=f"stop_loss_{i}")
    if symbol and qty > 0 and invested > 0:
        portfolio.append({"symbol": symbol, "qty": qty, "invested": invested, "stop_loss_pct": stop_loss_pct})

if not portfolio:
    st.warning("Add stocks to your portfolio to proceed.")
    st.stop()

# --- Tabs for Features ---
tabs = st.tabs(
    [
        "1️⃣ Portfolio Summary", "2️⃣ Predictions & Technical Indicators", "3️⃣ Sector Allocation",
        "4️⃣ Monte Carlo Simulation", "5️⃣ Trending Sectors & Stocks", "6️⃣ Investment Advice",
        "7️⃣ Profit/Loss Analysis", "8️⃣ Tax Impact", "9️⃣ Risk Management",
        "🔟 News Sentiment Analysis", "1️⃣1️⃣ Final Tip Sheet"
    ]
)

# --- Implement Features ---
with tabs[0]:
    st.header("1️⃣ Portfolio Summary")
    summary_data = []
    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            qty = stock["qty"]
            invested = stock["invested"]
            stop_loss = current_price * (1 - stock["stop_loss_pct"] / 100)
            current_value = qty * current_price
            profit_loss = current_value - invested
            summary_data.append({
                "Stock": stock["symbol"],
                "Invested (₹)": invested,
                "Current Price (₹)": current_price,
                "Current Value (₹)": current_value,
                "Profit/Loss (₹)": profit_loss,
                "Stop Loss (₹)": stop_loss
            })
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)
# --- Feature 2: Predictions & Technical Indicators ---
with tabs[1]:
    st.header("2️⃣ Predictions & Technical Indicators")

    for stock in portfolio:
        st.subheader(f"Prediction & Technical Indicators for {stock['symbol']}")

        # Fetch historical data
        data = fetch_stock_data(stock["symbol"], "6mo")

        if not data.empty:
            # --- Predictions (Prophet Model) ---
            forecast = predict_prices(data, forecast_days=30)

            # Ensure predictions start from today and align with trading days
            today = pd.Timestamp.now().normalize()
            trading_days = get_next_trading_days(today, num_days=5)

            # Filter predictions to only include trading days
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

            # Display next 5 days prediction
            st.write("### Next 5 Days Prediction")
            st.table(next_5_days)

            # --- Technical Indicators ---
            st.subheader("Technical Indicators")

            # Ensure sufficient data for indicator calculations
            if len(data) < 20:
                st.warning(f"Not enough data to calculate all indicators for {stock['symbol']}. At least 20 rows are required.")
                continue

            # Calculate Indicators
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

            # Display Values
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

            # Display Insights
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
    st.header("3️⃣ Sector Allocation")

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
    st.header("4️⃣ Monte Carlo Simulation")

    for stock in portfolio:
        st.subheader(f"Monte Carlo Simulation for {stock['symbol']}")

        # Fetch historical data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            # Run Monte Carlo Simulation
            simulations = monte_carlo_simulation(data, days=30, simulations=1000)

            # Calculate Statistical Metrics
            mean_price = np.mean(simulations)
            median_price = np.median(simulations)
            lower_ci = np.percentile(simulations, 2.5)  # 2.5th percentile for 95% confidence interval
            upper_ci = np.percentile(simulations, 97.5)  # 97.5th percentile for 95% confidence interval

            # Plot Histogram with Annotations
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

            # Display Metrics
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
    st.header("5️⃣ Trending Stocks and Sectors")

    # --- Market Movers Section ---
    st.subheader("📊 Market Movers (Trending Stocks)")

    # Fetch market movers
    market_movers_df = fetch_market_movers()

    if not market_movers_df.empty:
        # Group by category
        for category in market_movers_df["Category"].unique():
            st.subheader(category)  # Display category (e.g., Top Gainers, Top Losers)
            category_data = market_movers_df[market_movers_df["Category"] == category]

            # Create cards for each stock
            cols = st.columns(3)  # Display cards in a grid (3 per row)
            for index, row in category_data.iterrows():
                col = cols[index % 3]  # Distribute cards across columns
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
    st.subheader("📈 Trending Sectors")
    try:
        # Fetch sector performance
        sectors_df = fetch_sector_performance()

        if not sectors_df.empty:
            st.write("### Sector Performance (Last 5 Days)")
            st.dataframe(sectors_df.style.format({"Performance (%)": "{:.2f}%", "Current Price": "₹{:.2f}"}))

            # Create Grouped Bar Chart
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

# --- Feature 6: Investment Advice ---
with tabs[5]:
    st.header("6️⃣ Investment Advice")

    # Input: Total funds available for investment
    total_funds = st.sidebar.number_input("Enter total funds available for investment (₹):", min_value=500.0, step=100.0)

    # Generate portfolio advice and diversification options
    portfolio_advice, diversification_options = generate_portfolio_advice(portfolio, total_funds)

    # Display portfolio-specific advice
    st.subheader("Current Portfolio Advice (Buy/Hold/Sell)")
    if not portfolio_advice.empty:
        st.dataframe(portfolio_advice)
    else:
        st.warning("No advice available for your portfolio.")

    # Display diversification options
    st.subheader("Diversification Opportunities")
    if not diversification_options.empty:
        st.dataframe(diversification_options)
    else:
        st.warning("No diversification options available.")

    # Show recommendations for investment
    st.subheader("Investment Opportunities")
    if total_funds > 0:
        recommendations_df = recommend_profitable_stocks(total_funds)
        if not recommendations_df.empty:
            st.write("Recommended Stocks for Investment:")
            st.dataframe(recommendations_df)
        else:
            st.warning("No stocks available for recommendation.")
    else:
        st.warning("Please enter a valid total amount to see recommendations.")


# --- Updated Profit/Loss Analysis ---
with tabs[6]:
    st.header("7️⃣ Profit/Loss Analysis")

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

            # Calculate additional metrics
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

    # Display Metrics
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
    st.header("8️⃣ Tax Impact Analysis (Detailed)")

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
            holding_period = 6  # Placeholder: replace with actual holding period logic
            dividend_income = stock.get("dividend", 0)  # Assume user provides dividend income per stock

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

    # Calculate Total Tax
    total_tax = total_stcg_tax + total_ltcg_tax + total_dividend_tax
    total_tax_with_cess = total_tax * 1.04  # Adding 4% cess

    # Display Tax Breakdown
    tax_df = pd.DataFrame(tax_data)
    st.subheader("Detailed Tax Breakdown")
    st.dataframe(tax_df)

    # Display Metrics
    st.metric("Total Short-Term Capital Gains Tax (₹)", round(total_stcg_tax, 2))
    st.metric("Total Long-Term Capital Gains Tax (₹)", round(total_ltcg_tax, 2))
    st.metric("Total Dividend Tax (₹)", round(total_dividend_tax, 2))
    st.metric("Total Tax with Cess (₹)", round(total_tax_with_cess, 2))

    # Visualize Tax Distribution
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
    st.header("9️⃣ Risk Management (Detailed)")

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
        # Fetch historical data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            returns = data["Close"].pct_change().dropna()
            returns_data.append(returns)

            # Calculate VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            invested = stock["invested"]
            stock_var = invested * var
            portfolio_var += stock_var

            # Calculate CVaR
            cvar = returns[returns <= var].mean()
            stock_cvar = invested * cvar
            portfolio_cvar += stock_cvar

            # Simulated Beta (Placeholder)
            beta = 1.2  # Mock beta; integrate an API for actual beta values
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

    # Aggregate Risk Metrics
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
    downturn_scenario = -0.2  # Simulate a 20% market drop
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
    st.header("🔟 Enhanced News Sentiment Analysis")

    # --- Enhanced News Sentiment Analysis ---
with tabs[9]:
    st.header("🔟 News Sentiment Analysis & Recent Market News")

    # 1️⃣ User-Specific News Sentiments
    st.subheader("News Sentiments Related to Your Portfolio")
    sentiment_data = []
    sentiment_trends = []

    for stock in portfolio:
        # Fetch news articles for the stock
        news_articles = fetch_google_news(stock["symbol"])
        sentiment_scores = []

        for article in news_articles:
            # Perform sentiment analysis
            text_blob_analysis = TextBlob(article["title"])
            polarity = text_blob_analysis.sentiment.polarity
            subjectivity = text_blob_analysis.sentiment.subjectivity

            # Enhanced sentiment analysis using VADER
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

    # Display Portfolio Sentiments
    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df)

    # Visualize Portfolio Sentiment Distribution
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

    # 2️⃣ Recent General News
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
    st.header("1️⃣1️⃣ Final Tip Sheet")

    st.markdown("### Portfolio Summary")
    st.write(f"**Total Invested (₹):** {total_invested}")
    st.write(f"**Current Value (₹):** {total_current_value}")
    st.write(f"**Overall Profit/Loss (₹):** {total_current_value - total_invested}")
    st.write(f"**Portfolio ROI (%):** {round(((total_current_value - total_invested) / total_invested) * 100, 2)}")

    st.markdown("### Investment Recommendations")
    for index, row in portfolio_advice.iterrows():
        st.write(f"**{row['Stock']}:** {row['Action']} {row['Quantity']} shares. {row['Reason']}")

    st.markdown("### Risk Assessment")
    st.write(f"**Portfolio VaR (₹):** {round(portfolio_var, 2)}")
    st.write(f"**Portfolio CVaR (₹):** {round(portfolio_cvar, 2)}")
    st.write(f"**Portfolio Beta:** {round(portfolio_beta, 2)}")
    st.write(f"**Portfolio Sharpe Ratio:** {round(sharpe_ratio, 2)}")

    st.markdown("### Sector Insights")
    try:
        # Fetch sector performance (reuse or fetch again)
        if 'sectors_df' in locals() and not sectors_df.empty:
            top_sectors = sectors_df.head(3)  # Top 3 performing sectors
            st.write("Top Performing Sectors:")
            st.dataframe(top_sectors)
        else:
            st.warning("No sector performance data available.")
    except Exception as e:
        st.error(f"Error fetching sector insights: {e}")

    st.markdown("### News Sentiment Highlights")
    for index, row in sentiment_df.iterrows():
        st.write(f"**{row['Stock']}:** {row['Sentiment Label']} sentiment. "
                 f"Top Articles: {', '.join(row['Recent Articles'][:3])}")

    st.markdown("### Tax Impact")
    st.write(f"**Short-Term Capital Gains Tax (₹):** {round(total_stcg_tax, 2)}")
    st.write(f"**Long-Term Capital Gains Tax (₹):** {round(total_ltcg_tax, 2)}")
    st.write(f"**Dividend Tax (₹):** {round(total_dividend_tax, 2)}")
    st.write(f"**Total Tax Liability with Cess (₹):** {round(total_tax_with_cess, 2)}")
    st.markdown("**Tax Tips:**")
    st.write("- Use losses to offset gains where applicable.")
    st.write("- Ensure to leverage the ₹1,00,000 LTCG exemption.")
    st.write("- Declare losses to carry forward for up to 8 years.")

    st.markdown("### Future Predictions")
    st.write("Prophet-based predictions for key stocks in the portfolio:")
    for stock in portfolio:
        st.write(f"**{stock['symbol']}:**")
        if "Next 5 Days Prediction" in locals():
            st.table(next_5_days)  # Display next 5-day predictions

    st.markdown("### Stress Test Results")
    st.write(f"Total Portfolio Value after a 20% market downturn: ₹{total_current_value + cumulative_stress_loss:.2f}")

    st.markdown("### General Tips")
    st.write("- Monitor technical indicators and sentiment trends regularly.")
    st.write("- Diversify investments across sectors to reduce risk.")
    st.write("- Use Monte Carlo simulations and risk metrics for better planning.")
    st.write("- Consult a tax advisor for efficient tax management.")

    st.markdown("### Action Plan")
    st.write("1. Adjust portfolio based on diversification recommendations.")
    st.write("2. Execute buy/sell actions based on sentiment and technical indicators.")
    st.write("3. Plan for tax implications and optimize tax liability.")
    st.write("4. Continue monitoring market and sector performance weekly.")
