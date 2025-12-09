import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime

# 1: Load sentiment data to get tickers
print("Loading sentiment data...")
try:
    df_sentiment = pd.read_csv('wsb_daily_sentiment.csv')
    print(f"Loaded sentiment data with {len(df_sentiment)} rows")
except FileNotFoundError:
    print("Error: 'wsb_daily_sentiment.csv' file not found.")
    exit(1)
    
tickers = df_sentiment['ticker'].unique()
print(f"Found {len(tickers)} unique tickers")
print(f"Tickers: {', '.join(tickers[:10])}...")

# 2: function to download stock data

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a single ticker
    Returns DataFrame with date, close price, and returns
    """
    try:
        print(f"Downloading {ticker}...", end=" ")

        # Download data from Yahoo Finance
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock.empty:
            print("No data")
            return None

        # Reset index to make date a column
        stock = stock.reset_index()

        # Handle multi-level columns (yfinance sometimes returns these)
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        # Create simple DataFrame with needed columns
        df = pd.DataFrame({
            'date': pd.to_datetime(stock['Date']).dt.date,
            'ticker': ticker,
            'close': stock['Close'].values.flatten() if hasattr(stock['Close'].values, 'flatten') else stock['Close'].values,
            'volume': stock['Volume'].values.flatten() if hasattr(stock['Volume'].values, 'flatten') else stock['Volume'].values
        })

        # Calculate daily returns (percent change)
        df['return_1d'] = df['close'].pct_change() * 100

        # Calculate forward returns (what we want to predict)
        df['forward_return_1d'] = df['return_1d'].shift(-1)  # Next day return
        df['forward_return_3d'] = ((df['close'].shift(-3) / df['close']) - 1) * 100  # 3-day return
        df['forward_return_7d'] = ((df['close'].shift(-7) / df['close']) - 1) * 100  # 7-day return

        print(f"Success ({len(df)} days)")
        return df

    except Exception as e:
        print(f"Error: {e}")
        return None
    
# 3: Set date range
# Get date range
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
min_date = df_sentiment['date'].min()
max_date = df_sentiment['date'].max()

# Add buffer for forward returns
start_date = min_date - timedelta(days=30)
end_date = max_date + timedelta(days=14)

print(f"\nDownloading stock data from {start_date} to {end_date}")

# 4: Download data for all tickers
all_stock_data = []

for ticker in tickers:
    stock_df = download_stock_data(ticker, start_date, end_date)
    if stock_df is not None:
        all_stock_data.append(stock_df)

# 5: Combine all stock data
if all_stock_data:
    df_stocks = pd.concat(all_stock_data, ignore_index=True)
    print(f"\nTotal stock data collected: {len(df_stocks)} rows")

    # Show sample
    print("\nSample stock data:")
    print(df_stocks.head(10))

    # Show summary
    print("\nSummary by ticker:")
    print(df_stocks.groupby('ticker')['close'].count().sort_values(ascending=False).head(10))

    # Save to CSV
    df_stocks.to_csv('stock_prices.csv', index=False)
    print("\nStock data saved to stock_prices.csv")
else:
    print("No stock data collected!")