# Step 1: Collecting Reddit Data from r/WallStreetBets
# Install required packages first:
# pip install praw pandas

import praw
import pandas as pd
import re
from datetime import datetime

# ===== CONFIGURATION =====
# You'll need to get these from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "your_client_id_here"
REDDIT_CLIENT_SECRET = "your_client_secret_here"
REDDIT_USER_AGENT = "WSB_Sentiment_Analysis_v1.0"

# ===== STEP 1: Connect to Reddit =====
print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# ===== STEP 2: Function to extract stock tickers from text =====
def extract_tickers(text):
    """
    Find all stock tickers in the format $TICKER (e.g., $AAPL, $GME)
    Returns a list of unique tickers
    """
    if text is None:
        return []
    
    # Find all words starting with $ followed by 1-5 capital letters
    tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    
    # Return unique tickers only
    return list(set(tickers))

# ===== STEP 3: Collect posts from WSB =====
def collect_wsb_posts(limit=100):
    """
    Collect recent posts from r/WallStreetBets
    Returns a pandas DataFrame
    """
    print(f"Collecting {limit} posts from r/WallStreetBets...")
    
    subreddit = reddit.subreddit("wallstreetbets")
    
    posts_data = []
    
    # Get recent posts
    for post in subreddit.new(limit=limit):
        # Combine title and body for ticker extraction
        full_text = (post.title or "") + " " + (post.selftext or "")
        
        # Extract tickers mentioned
        tickers = extract_tickers(full_text)
        
        # Only keep posts that mention at least one ticker
        if tickers:
            post_info = {
                'date': datetime.fromtimestamp(post.created_utc).date(),
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'tickers': ','.join(tickers),  # Store as comma-separated string
                'text': full_text[:200]  # First 200 chars for reference
            }
            posts_data.append(post_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(posts_data)
    print(f"Collected {len(df)} posts with ticker mentions")
    
    return df

# ===== STEP 4: Run the collection =====
if __name__ == "__main__":
    # Collect posts
    df_posts = collect_wsb_posts(limit=500)
    
    # Display first few rows
    print("\nFirst 5 posts:")
    print(df_posts.head())
    
    # Show ticker distribution
    print("\nMost mentioned tickers:")
    all_tickers = df_posts['tickers'].str.split(',').explode()
    print(all_tickers.value_counts().head(10))
    
    # Save to CSV
    df_posts.to_csv('wsb_posts.csv', index=False)
    print("\nData saved to wsb_posts.csv")