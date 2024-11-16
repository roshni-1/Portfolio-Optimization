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

# --- Implemented Features ---
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
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            simulations = monte_carlo_simulation(data, days=30, simulations=1000)
            fig = px.histogram(
                simulations, nbins=50, title=f"Monte Carlo Simulation for {stock['symbol']}",
                labels={"value": "Simulated Price (₹)", "count": "Frequency"}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

            st.markdown(f"""
                **What this means for {stock['symbol']}:**  
                - This histogram shows potential price ranges for {stock['symbol']} in the next 30 days.  
                - Wider distributions indicate higher volatility, meaning the price could vary more significantly.  
            """)

# --- Feature 5: Trending Sectors & Stocks ---
with tabs[4]:
    st.header("5️⃣ Trending Sectors & Stocks")

    # Fetch real-time NIFTY 50 symbols
    nifty_50_symbols = fetch_nifty_50_symbols()

    sector_performance = {}
    trending_stocks = []

    for symbol in nifty_50_symbols:
        try:
            ticker = yf.Ticker(symbol)
            stock_info = ticker.info
            sector = stock_info.get("sector", "Unknown")
            data = ticker.history(period="1d", interval="1m")

            if not data.empty:
                open_price = data["Open"].iloc[0]
                close_price = data["Close"].iloc[-1]
                percent_change = ((close_price - open_price) / open_price) * 100

                if sector != "Unknown":
                    if sector not in sector_performance:
                        sector_performance[sector] = []
                    sector_performance[sector].append(percent_change)

                if abs(percent_change) > 2:
                    trending_stocks.append({
                        "Stock": symbol,
                        "Sector": sector,
                        "Current Price (₹)": round(close_price, 2),
                        "% Change": round(percent_change, 2)
                    })
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")

    avg_sector_performance = {sector: np.mean(changes) for sector, changes in sector_performance.items()}
    if avg_sector_performance:
        sector_df = pd.DataFrame(list(avg_sector_performance.items()), columns=["Sector", "Average % Change"])
        fig = px.bar(sector_df, x="Sector", y="Average % Change", title="Trending Sectors")
        st.plotly_chart(fig)
    else:
        st.write("No trending sectors found.")

    if trending_stocks:
        trending_stocks_df = pd.DataFrame(trending_stocks)
        st.dataframe(trending_stocks_df)
    else:
        st.write("No trending stocks found.")


# --- Feature 6: Investment Advice ---
with tabs[5]:
    st.header("6️⃣ Investment Advice")

    advice = []
    for stock in portfolio:
        data = fetch_stock_data(stock["symbol"], "6mo")
        if not data.empty:
            current_price = data["Close"].iloc[-1]
            moving_avg = data["Close"].rolling(window=20).mean().iloc[-1]

            if current_price > moving_avg:
                advice.append((stock["symbol"], "Consider buying, upward trend"))
            else:
                advice.append((stock["symbol"], "Consider selling, downward trend"))

    advice_df = pd.DataFrame(advice, columns=["Stock", "Advice"])
    st.dataframe(advice_df)

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

