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
