import http.client
import json
import streamlit as st
import pandas as pd

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

# --- Display Market Movers as Cards ---
st.header("📊 Market Movers (Trending Stocks)")

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
