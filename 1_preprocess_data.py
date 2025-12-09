import pandas as pd
from textblob import TextBlob

# 1: load data
print("Loading Reddit posts...")    
try:    
    df_posts = pd.read_csv('wsb_posts.csv')
    print(f"Loaded {len(df_posts)} posts")
except FileNotFoundError:
    print("Error: 'wsb_posts.csv' file not found.")
    exit(1)

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
