import numpy as np
import pandas as pd
import yfinance as yf

# Function to fetch current price for each stock with error handling
def get_current_price(stock_symbols):
    current_prices = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        # Check if data is available
        if not data.empty:
            current_prices[symbol] = data['Close'].iloc[-1]  # Get the latest close price
        else:
            print(f"Warning: No price data found for {symbol}. It may be delisted or have no recent trading data.")
            current_prices[symbol] = None  # Set to None if no data is available
    
    return current_prices

# Function to calculate profit and loss for each stock
def calculate_profit_loss(stock_symbols, quantities, purchase_prices):
    current_prices = get_current_price(stock_symbols)
    profit_loss_summary = {}
    
    for symbol in stock_symbols:
        purchase_price = purchase_prices[symbol]
        quantity = quantities[symbol]
        current_price = current_prices[symbol]
        
        # Skip calculation if current price data is unavailable
        if current_price is None:
            print(f"Skipping {symbol} due to lack of price data.")
            continue
        
        # Calculate profit or loss
        profit_loss = (current_price - purchase_price) * quantity
        profit_loss_percentage = ((current_price - purchase_price) / purchase_price) * 100
        
        # Determine status and action
        status = "Profit" if profit_loss > 0 else "Loss" if profit_loss < 0 else "Break-even"
        
        profit_loss_summary[symbol] = {
            "Quantity": quantity,
            "Purchase Price": purchase_price,
            "Current Price": current_price,
            "Profit/Loss": profit_loss,
            "Profit/Loss (%)": profit_loss_percentage,
            "Status": status
        }
        
    return profit_loss_summary

# Main function to demonstrate profit and loss tracking
def main():
    # Gather user input for stock symbols, quantities, and purchase prices
    input_symbols = input("Enter stock symbols with suffix (e.g., TATASTEEL.NS, RPOWER.BS): ")
    stock_symbols = [symbol.strip() for symbol in input_symbols.split(",")]
    quantities = {}
    purchase_prices = {}
    for symbol in stock_symbols:
        quantities[symbol] = int(input(f"Enter quantity for {symbol}: "))
        purchase_prices[symbol] = float(input(f"Enter purchase price for {symbol}: "))
    
    # Calculate and display profit or loss for each stock
    profit_loss_summary = calculate_profit_loss(stock_symbols, quantities, purchase_prices)
    
    print("\nProfit and Loss Summary:")
    for symbol, summary in profit_loss_summary.items():
        print(f"\nStock: {symbol}")
        print(f"  Quantity: {summary['Quantity']}")
        print(f"  Purchase Price: {summary['Purchase Price']:.2f} INR")
        print(f"  Current Price: {summary['Current Price']:.2f} INR")
        print(f"  Status: {summary['Status']}")
        print(f"  Profit/Loss: {summary['Profit/Loss']:.2f} INR ({summary['Profit/Loss (%)']:.2f}%)")
        
        # Provide advice based on the profit/loss status
        if summary["Status"] == "Profit":
            print("  Advice: Consider holding or selling a portion to lock in some gains.")
        elif summary["Status"] == "Loss":
            print("  Advice: Consider holding if the stock has potential for recovery, or sell to cut losses.")
        else:
            print("  Advice: No gain or loss at the moment, monitor performance before deciding.")
            
if __name__ == "__main__":
    main()
