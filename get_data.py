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
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    print("Connected to Reddit successfully.")
except Exception as e:
    print(f"Failed to connect to Reddit: {e}")
    exit(1)
    
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
    
    posts_to_collect = 10000
    df_posts = collect_wsb_posts(limit=posts_to_collect)

    # Display some sample data
    print("\nFirst 5 posts:")
    print(df_posts.head())

    # Display most mentioned tickers
    print("\nMost mentioned tickers:")
    all_tickers = df_posts['tickers'].str.split(',').explode()
    print(all_tickers.value_counts().head(10))

    # Save to CSV
    df_posts.to_csv('wsb_posts.csv', index=False)
    print("\nData saved to wsb_posts.csv")
    
    