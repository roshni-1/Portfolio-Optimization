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
    """Fetch sector performance in Indian market using yfinance."""
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
def investment_advice(portfolio, total_funds):
    """
    Generate investment advice based on the user's portfolio, available funds, and market conditions.
    
    Parameters:
    - portfolio: List of dictionaries containing stock symbols, quantities, invested amounts, and stop-loss percentages.
    - total_funds: Total funds available for investment.
    
    Returns:
    - advice_df: DataFrame containing buy/hold/sell advice and quantities.
    - diversification_options: Suggestions for other investment opportunities.
    - explainability: Detailed explanations for the advice.
    """
    advice_data = []
    diversification_options = []
    explainability = []

    for stock in portfolio:
        # Fetch current stock data
        data = fetch_stock_data(stock["symbol"], "6mo")
        if data.empty:
            st.error(f"Unable to fetch data for {stock['symbol']}. Skipping.")
            continue

        # Extract necessary data
        current_price = data["Close"].iloc[-1]
        qty_owned = stock["qty"]
        invested_amount = stock["invested"]
        stop_loss_price = current_price * (1 - stock["stop_loss_pct"] / 100)

        # Determine buy/hold/sell action
        action = ""
        quantity = 0

        if current_price < stop_loss_price:  # Stock is nearing stop-loss
            action = "Sell"
            quantity = qty_owned
            explainability.append(
                f"{stock['symbol']} is nearing its stop-loss price of ₹{stop_loss_price:.2f}. "
                f"Consider selling all holdings."
            )
        elif current_price < invested_amount / qty_owned:  # Current price is below average purchase price
            action = "Hold"
            explainability.append(
                f"{stock['symbol']} is trading below your average purchase price. "
                f"Consider holding for recovery."
            )
        else:  # Potential buy opportunity
            max_buy_qty = total_funds // current_price
            action = "Buy"
            quantity = int(max_buy_qty * 0.2)  # Invest 20% of available funds
            explainability.append(
                f"{stock['symbol']} is performing well, with potential for growth. "
                f"Consider buying {quantity} shares."
            )

        advice_data.append({
            "Stock": stock["symbol"],
            "Current Price (₹)": current_price,
            "Action": action,
            "Quantity": quantity,
        })

    # Generate diversification options based on sector performance
    sectors_df = fetch_sector_performance()
    if not sectors_df.empty:
        top_sectors = sectors_df.head(3)  # Top 3 performing sectors
        for _, row in top_sectors.iterrows():
            diversification_options.append({
                "Sector": row["Sector"],
                "Performance (%)": row["Performance (%)"],
                "Suggested Investment (₹)": total_funds * 0.1  # Suggest 10% of funds for diversification
            })
        explainability.append(
            "Consider diversifying into top-performing sectors for balanced growth."
        )

    # Create DataFrames for advice and diversification options
    advice_df = pd.DataFrame(advice_data)
    diversification_df = pd.DataFrame(diversification_options)

    return advice_df, diversification_df, explainability

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

# --- Feature 5: Trending Sectors & Stocks ---
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

# --- Feature 6: Investment Advice ---
with tabs[5]:
    st.header("6️⃣ Investment Advice")

   # Fetch total funds available for investment
    total_funds = st.sidebar.number_input("Total Funds Available (₹)", min_value=1000.0, step=100.0)

    # Generate investment advice
    advice_df, diversification_df, explainability = investment_advice(portfolio, total_funds)

    # Display investment advice
    st.subheader("Investment Advice (Buy/Hold/Sell)")
    st.dataframe(advice_df)

    # Display diversification options
    st.subheader("Other Investment Opportunities")
    if not diversification_df.empty:
        st.dataframe(diversification_df)
    else:
        st.warning("No diversification options available.")

    # Display explainability
    st.subheader("Explainability")
    for explanation in explainability:
        st.write(f"- {explanation}")

# --- Feature 7: Profit/Loss Analysis ---
with tabs[6]:
    st.header("7️⃣ Profit/Loss Analysis")

    total_invested = sum(stock["invested"] for stock in portfolio)
    total_current_value = 0
    profit_loss_data = []

    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_value = stock["qty"] * data["Close"].iloc[-1]
            total_current_value += current_value
            profit_loss_data.append({
                "Stock": stock["symbol"],
                "Invested (₹)": stock["invested"],
                "Current Value (₹)": current_value,
                "Profit/Loss (₹)": current_value - stock["invested"]
            })

    profit_loss_df = pd.DataFrame(profit_loss_data)
    st.dataframe(profit_loss_df)
    st.metric("Total Invested (₹)", total_invested)
    st.metric("Total Current Value (₹)", total_current_value)
    st.metric("Overall Profit/Loss (₹)", total_current_value - total_invested)

# --- Feature 8: Tax Impact ---
with tabs[7]:
    st.header("8️⃣ Tax Impact")

    tax_data = []
    total_tax = 0

    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            invested = stock["invested"]
            qty = stock["qty"]
            profit = (current_price * qty) - invested
            holding_period = 6  # Mocked as 6 months; use actual dates for real calculation.

            if profit > 0:
                if holding_period < 12:
                    tax = profit * 0.15  # Short-Term Capital Gains Tax
                else:
                    taxable_gain = max(0, profit - 100000)  # Long-Term Capital Gains Exemption
                    tax = taxable_gain * 0.10
                total_tax += tax

                tax_data.append({
                    "Stock": stock["symbol"],
                    "Profit (₹)": profit,
                    "Tax Liability (₹)": tax,
                    "Holding Period (Months)": holding_period
                })
            else:
                tax_data.append({
                    "Stock": stock["symbol"],
                    "Profit (₹)": profit,
                    "Tax Liability (₹)": 0,
                    "Holding Period (Months)": holding_period
                })

    tax_df = pd.DataFrame(tax_data)
    st.dataframe(tax_df)
    st.metric("Total Tax Liability (₹)", round(total_tax, 2))

# --- Feature 9: Risk Management ---
with tabs[8]:
    st.header("9️⃣ Risk Management (VaR)")

    confidence_level = 0.95
    portfolio_var = 0
    var_data = []

    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            returns = data["Close"].pct_change().dropna()
            var = np.percentile(returns, (1 - confidence_level) * 100)
            invested = stock["invested"]
            stock_var = invested * var
            portfolio_var += stock_var

            var_data.append({
                "Stock": stock["symbol"],
                "Invested (₹)": invested,
                "VaR (₹)": round(stock_var, 2)
            })

    var_df = pd.DataFrame(var_data)
    st.dataframe(var_df)
    st.metric("Portfolio VaR (₹)", round(portfolio_var, 2))

# --- Feature 10: News Sentiment Analysis ---
with tabs[9]:
    st.header("🔟 News Sentiment Analysis")
    sentiment_data = []

    for stock in portfolio:
        news_articles = fetch_google_news(stock["symbol"])
        sentiment_scores = []

        for article in news_articles:
            analysis = TextBlob(article["title"])
            sentiment_scores.append(analysis.sentiment.polarity)

        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment = "Positive" if avg_sentiment > 0 else "Negative"
        else:
            avg_sentiment = 0
            sentiment = "Neutral"

        sentiment_data.append({
            "Stock": stock["symbol"],
            "Sentiment": sentiment,
            "Average Score": round(avg_sentiment, 2),
            "Articles": [article["title"] for article in news_articles]
        })

    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df)

# --- Feature 11: Final Tip Sheet ---
with tabs[10]:
    st.header("1️⃣1️⃣ Final Tip Sheet")
    st.markdown("### Key Takeaways:")
    st.write("1. Diversify your investments across sectors to manage risks.")
    st.write("2. Monitor trending stocks and sectors for new opportunities.")
    st.write("3. Use Monte Carlo simulations to assess potential outcomes.")
    st.write("4. Review tax impacts to minimize liabilities.")
    st.write("5. Apply insights from technical analysis and news sentiment.")