import http.client
import json
import urllib.parse
import streamlit as st
import pandas as pd

# --- Fetch Stock Logo Using Google Search72 API ---
def fetch_stock_logo(stock_symbol):
    """Fetch a stock's logo using the Google Search72 API."""
    conn = http.client.HTTPSConnection("google-search72.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5d63bb22bemshb6e582f5cdfd2cdp1d4344jsn14c1f5b16633",
        'x-rapidapi-host': "google-search72.p.rapidapi.com"
    }

    try:
        # Encode the query string properly
        query = urllib.parse.quote(f"LOGO {stock_symbol}")
        conn.request("GET", f"/imagesearch?q={query}&gl=us&lr=lang_en&num=1&start=0", headers=headers)
        res = conn.getresponse()
        data = res.read()
        response_json = json.loads(data.decode("utf-8"))

        # Extract the first image URL if available
        image_results = response_json.get("items", [])
        if image_results:
            return image_results[0].get("link", "https://via.placeholder.com/50?text=Logo")

        # Fallback to placeholder if no image is found
        return "https://via.placeholder.com/50?text=Logo"

    except Exception as e:
        st.error(f"Error fetching logo for {stock_symbol}: {e}")
        return "https://via.placeholder.com/50?text=Logo"

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
            for stock in category.get("quotes", []):
                stock_symbol = stock.get("symbol", "N/A")
                price = stock.get("regularMarketPrice", "N/A")

                # Fetch logo using Google Search72 API
                logo_url = fetch_stock_logo(stock_symbol)

                movers.append({
                    "Symbol": stock_symbol,
                    "Price (₹)": price,
                    "Logo": logo_url
                })

        return pd.DataFrame(movers)

    except Exception as e:
        st.error(f"Error fetching market movers: {e}")
        return pd.DataFrame()

# --- Display Market Movers ---
st.header("📊 Market Movers (Trending Stocks)")

# Fetch market movers
market_movers_df = fetch_market_movers()

if not market_movers_df.empty:
    st.write("### Top Market Movers")

    # Display the data in a table format
    for index, row in market_movers_df.iterrows():
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.image(row["Logo"], width=50)  # Logo from Google Search72 API
        with col2:
            st.markdown(f"**{row['Symbol']}**")  # Stock symbol
        with col3:
            st.markdown(f"**₹{row['Price (₹)']}**")  # Stock price

    # Add a download button to export the data
    st.download_button(
        label="Download as CSV",
        data=market_movers_df.to_csv(index=False),
        file_name="market_movers.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download as Excel",
        data=market_movers_df.to_excel(index=False),
        file_name="market_movers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("No market movers data available. Try again later.")
