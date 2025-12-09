# dsproject.py

import praw
import pandas as pd
import re
from datetime import datetime


# Information get from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "DfeJt179IwKBJh5IKIOpYQ"
REDDIT_CLIENT_SECRET = "ANZEcmLDpVf1eSzDE0hogh6DD88vIw"
REDDIT_USER_AGENT = "WSB_Sentiment_Analysis_v1.0"

# 1: Connect to Reddit
print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# 2: Function to extract stock tickers from text
def extract_tickers(text):
    if text is None:
        return []

    # Find all words starting with $ followed by 1-5 capital letters
    tickers = re.findall(r'\$([A-Z]{1,5})\b', text)

    # Return unique tickers only
    return list(set(tickers))

# 3: Collect posts from WSB
def collect_wsb_posts(limit=100):
    print(f"Collecting {limit} posts from r/WallStreetBets...")

    subreddit = reddit.subreddit("wallstreetbets")

    posts_data = []

    # Get recent posts
    for post in subreddit.new(limit=limit):
        full_text = (post.title or "") + " " + (post.selftext or "")
        tickers = extract_tickers(full_text)
        if tickers:
            post_info = {
                'date': datetime.fromtimestamp(post.created_utc).date(),
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'tickers': ','.join(tickers),
                'text': full_text[:200]
            }
            posts_data.append(post_info)

    # Convert to DataFrame
    df = pd.DataFrame(posts_data)
    print(f"Collected {len(df)} posts with ticker mentions")

    return df

# 4: Run the collection
if __name__ == "__main__":
    df_posts = collect_wsb_posts(limit=10000)

    print("\nFirst 5 posts:")
    print(df_posts.head())

    print("\nMost mentioned tickers:")
    all_tickers = df_posts['tickers'].str.split(',').explode()
    print(all_tickers.value_counts().head(10))

    df_posts.to_csv('wsb_posts.csv', index=False)
    print("\nData saved to wsb_posts.csv")

import pandas as pd
from textblob import TextBlob

# 1: Load the posts data
print("Loading Reddit posts...")
df_posts = pd.read_csv('wsb_posts.csv')

# 2: Simple sentiment analysis function
def calculate_sentiment(text):
    if pd.isna(text) or text == "":
        return 0

    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except:
        return 0

# 3: Add sentiment scores to posts
print("Calculating sentiment for each post...")
df_posts['sentiment'] = df_posts['text'].apply(calculate_sentiment)

# 4: Aggregate daily sentiment by ticker
print("Aggregating daily sentiment by ticker...")

# Split tickers
posts_expanded = []

for _, row in df_posts.iterrows():
    tickers = row['tickers'].split(',')
    for ticker in tickers:
        posts_expanded.append({
            'date': row['date'],
            'ticker': ticker,
            'sentiment': row['sentiment'],
            'score': row['score'],
            'num_comments': row['num_comments']
        })

df_expanded = pd.DataFrame(posts_expanded)

# 5: Create daily aggregated metrics
df_daily = df_expanded.groupby(['date', 'ticker']).agg({
    'sentiment': 'mean',           # Average sentiment
    'score': 'sum',                # Total upvotes
    'num_comments': 'sum',         # Total comments
    'ticker': 'count'              # Number of mentions
}).rename(columns={'ticker': 'mention_count'})

df_daily = df_daily.reset_index()

# 6: Create engagement metric
df_daily['engagement'] = (df_daily['score'] + df_daily['num_comments'] * 2) / df_daily['mention_count']

print("\nDaily sentiment data:")
print(df_daily.head(10))

# 7: Show summary statistics
print("\nSummary by ticker:")
ticker_summary = df_daily.groupby('ticker').agg({
    'sentiment': 'mean',
    'mention_count': 'sum',
    'engagement': 'mean'
}).sort_values('mention_count', ascending=False)

print(ticker_summary.head(10))

# 8: Save processed data
df_daily.to_csv('wsb_daily_sentiment.csv', index=False)
print("\nDaily sentiment data saved to wsb_daily_sentiment.csv")

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 1: Load sentiment data to get tickers
print("Loading sentiment data...")
df_sentiment = pd.read_csv('wsb_daily_sentiment.csv')

tickers = df_sentiment['ticker'].unique()
print(f"Found {len(tickers)} unique tickers")
print(f"Tickers: {', '.join(tickers[:10])}...")

# 2: Function to download stock data
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1: Load data
print("Loading data...")
df_sentiment = pd.read_csv('wsb_daily_sentiment.csv')
df_stocks = pd.read_csv('stock_prices.csv')

# Convert dates
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_stocks['date'] = pd.to_datetime(df_stocks['date'])

# =2: Merge sentiment with stock data
print("Merging sentiment and stock data...")
df_merged = pd.merge(
    df_sentiment,
    df_stocks,
    on=['date', 'ticker'],
    how='inner'
)

print(f"Merged dataset has {len(df_merged)} rows")
print(f"Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")

# Remove rows with missing forward returns
df_merged = df_merged.dropna(subset=['forward_return_1d', 'forward_return_3d', 'forward_return_7d'])

print(f"After removing missing values: {len(df_merged)} rows")

# 3: Calculate correlations
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

def calculate_correlation(df, x_col, y_col):
    # Remove any remaining NaN values
    valid_data = df[[x_col, y_col]].dropna()

    if len(valid_data) < 10:
        return None, None

    correlation, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])
    return correlation, p_value

# Correlations for different horizons
horizons = {
    '1-Day': 'forward_return_1d',
    '3-Day': 'forward_return_3d',
    '7-Day': 'forward_return_7d'
}

print("\n1. SENTIMENT vs FORWARD RETURNS")
print("-" * 60)
for horizon_name, return_col in horizons.items():
    corr, p_val = calculate_correlation(df_merged, 'sentiment', return_col)
    if corr is not None:
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{horizon_name:10} | Correlation: {corr:7.4f} | p-value: {p_val:.4f} {significance}")

print("\n2. MENTION COUNT vs FORWARD RETURNS")
print("-" * 60)
for horizon_name, return_col in horizons.items():
    corr, p_val = calculate_correlation(df_merged, 'mention_count', return_col)
    if corr is not None:
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{horizon_name:10} | Correlation: {corr:7.4f} | p-value: {p_val:.4f} {significance}")

print("\n3. ENGAGEMENT vs FORWARD RETURNS")
print("-" * 60)
for horizon_name, return_col in horizons.items():
    corr, p_val = calculate_correlation(df_merged, 'engagement', return_col)
    if corr is not None:
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{horizon_name:10} | Correlation: {corr:7.4f} | p-value: {p_val:.4f} {significance}")

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# 4: Create visualization
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('WSB Sentiment vs Stock Returns', fontsize=16, fontweight='bold')

# Plot sentiment vs returns for each horizon
for idx, (horizon_name, return_col) in enumerate(horizons.items()):
    ax = axes[0, idx]

    # Scatter plot
    ax.scatter(df_merged['sentiment'], df_merged[return_col], alpha=0.3, s=20)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel(f'{horizon_name} Return (%)')
    ax.set_title(f'Sentiment vs {horizon_name} Return')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

# Plot mention count vs returns
for idx, (horizon_name, return_col) in enumerate(horizons.items()):
    ax = axes[1, idx]

    ax.scatter(df_merged['mention_count'], df_merged[return_col], alpha=0.3, s=20)
    ax.set_xlabel('Mention Count')
    ax.set_ylabel(f'{horizon_name} Return (%)')
    ax.set_title(f'Mentions vs {horizon_name} Return')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to correlation_analysis.png")

# 5: Save merged data for modeling
df_merged.to_csv('merged_data.csv', index=False)
print("\nMerged data saved to merged_data.csv")

# 6: Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print("\nSentiment Statistics:")
print(df_merged['sentiment'].describe())

print("\nReturn Statistics:")
for horizon_name, return_col in horizons.items():
    print(f"\n{horizon_name} Returns:")
    print(df_merged[return_col].describe())

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1: Load merged data
print("Loading merged data...")
df = pd.read_csv('merged_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort by date (important for time series)
df = df.sort_values('date').reset_index(drop=True)

print(f"Total samples: {len(df)}")

# 2: features (X) and targets (y)
features = ['sentiment', 'mention_count', 'engagement']
X = df[features].values

# Remove any rows with NaN in features
valid_idx = ~np.isnan(X).any(axis=1)
X = X[valid_idx]
df_valid = df[valid_idx].reset_index(drop=True)

print(f"Valid samples after removing NaN: {len(df_valid)}")

# 3: Time Series Cross-Validation Setup
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

print(f"\nUsing {n_splits}-fold Time Series Cross-Validation")

# 4: MODEL 1 - Linear Regression (Predicting Returns)
print("\n" + "="*70)
print("MODEL 1: LINEAR REGRESSION - PREDICTING STOCK RETURNS")
print("="*70)

horizons = {
    '1-Day': 'forward_return_1d',
    '3-Day': 'forward_return_3d',
    '7-Day': 'forward_return_7d'
}

regression_results = {}

for horizon_name, target_col in horizons.items():
    print(f"\n{horizon_name} Horizon:")
    print("-" * 70)

    # Get target variable
    y = df_valid[target_col].values

    # Remove rows where target is NaN
    valid_target_idx = ~np.isnan(y)
    X_valid = X[valid_target_idx]
    y_valid = y[valid_target_idx]

    # Store results for each fold
    mse_scores = []
    r2_scores = []

    # Time series cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_valid), 1):
        # Split data
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"  Fold {fold}: MSE={mse:.4f}, R²={r2:.4f}")

    # Average results
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\n  Average MSE: {avg_mse:.4f}")
    print(f"  Average R²:  {avg_r2:.4f}")

    regression_results[horizon_name] = {
        'mse': avg_mse,
        'r2': avg_r2
    }

# 5: MODEL 2 - Logistic Regression
print("\n" + "="*70)
print("MODEL 2: LOGISTIC REGRESSION - PREDICTING RETURN DIRECTION")
print("="*70)

classification_results = {}

for horizon_name, target_col in horizons.items():
    print(f"\n{horizon_name} Horizon:")
    print("-" * 70)

    # Get target variable
    y_returns = df_valid[target_col].values

    # Convert to binary: 1 if positive return, 0 if negative
    y_direction = (y_returns > 0).astype(int)

    # Remove rows where target is NaN
    valid_target_idx = ~np.isnan(y_returns)
    X_valid = X[valid_target_idx]
    y_valid = y_direction[valid_target_idx]

    # Store results
    accuracy_scores = []

    # Time series cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_valid), 1):
        # Split data
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)

        print(f"  Fold {fold}: Accuracy={acc:.4f} ({acc*100:.2f}%)")

    # Average results
    avg_accuracy = np.mean(accuracy_scores)

    print(f"\n  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"  Baseline (random): 50.00%")

    classification_results[horizon_name] = {
        'accuracy': avg_accuracy
    }

# 6: Summary Visualization
print("\nCreating results visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R² scores
ax1 = axes[0]
horizons_list = list(regression_results.keys())
r2_values = [regression_results[h]['r2'] for h in horizons_list]

ax1.bar(horizons_list, r2_values, color=['#3498db', '#2ecc71', '#e74c3c'])
ax1.set_ylabel('R² Score')
ax1.set_title('Linear Regression Performance\n(Predicting Returns)', fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([min(r2_values) - 0.05, max(max(r2_values), 0) + 0.05])

# Plot 2: Accuracy scores
ax2 = axes[1]
accuracy_values = [classification_results[h]['accuracy'] * 100 for h in horizons_list]

ax2.bar(horizons_list, accuracy_values, color=['#3498db', '#2ecc71', '#e74c3c'])
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Logistic Regression Performance\n(Predicting Direction)', fontweight='bold')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Baseline (50%)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([45, max(accuracy_values) + 5])

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("Saved visualization to model_performance.png")

# 7: Save results summary
results_summary = {
    'Regression (R²)': [regression_results[h]['r2'] for h in horizons_list],
    'Classification (Accuracy)': [classification_results[h]['accuracy'] for h in horizons_list]
}

df_results = pd.DataFrame(results_summary, index=horizons_list)
df_results.to_csv('model_results.csv')
print("\nResults saved to model_results.csv")

print("\n" + "="*70)
print("MODELING COMPLETE!")
print("="*70)