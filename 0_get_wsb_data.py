import praw
import pandas as pd
import re
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Validate that credentials are provided
if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
    print("Error: Reddit API credentials not found in .env file")
    print("Please create a .env file with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
    exit(1)

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
    
    
# 2: Load list of stock tickers
print("Loading stock ticker list...")
try:
    # read ticker data from text file as dictionary
    ticker_file = 'yhallsym.txt'
    # read and parse the dictionary
    with open(ticker_file, 'r', encoding='utf-8') as f:
        content = f.read()
        ticker_dict = eval(content)  # Convert string representation to dict
        
        # filter for only tickers with valid company names
        ticker_dict = {k: v for k, v in ticker_dict.items() if v is not None}
        # filter for only tickers with 1-5 uppercase letters
        ticker_dict = {k: v for k, v in ticker_dict.items() if re.match(r'^[A-Z]{1,5}$', k)}
        # remove 'AONE' and 'TWOA' as they create weird artifacts
        ticker_dict.pop('AONE', None)
        ticker_dict.pop('TWOA', None)
        # strip company names past first comma
        ticker_dict = {k: v.split(',')[0] for k, v in ticker_dict.items()}
        # strip 'co', 'corp', 'inc', 'ltd', 'corporation' from company names
        ticker_dict = {k: re.sub(r'\b(co|corp|inc|ltd|corporation)\b', '', v, flags=re.IGNORECASE).strip() for k, v in ticker_dict.items()}
        
        universe_tickers = set(ticker_dict.keys())  # Extract just the ticker symbols
    print(f"Loaded {len(universe_tickers)} stock tickers.")
    print("Sample tickers:", list(universe_tickers)[:10])
except FileNotFoundError:
    print("Error: 'yhallsym.txt' file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading ticker file: {e}")
    exit(1)

# 3: Function to extract stock tickers from text
def extract_tickers(text):
    if text is None:
        return []

    # Find all words starting with $ followed by 1-5 capital letters
    tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    
    if not tickers:
        # Fallback: capitalized words if in universe
        tickers = re.findall(r'\b([A-Z]{1,5})\b', text) if re.search(r'\b([A-Z]{1,5})\b', text) in universe_tickers else []

    if not tickers:
        # Fallback: Company names mapped to tickers using ticker_dict
        for ticker, company in ticker_dict.items():
            if company != None:
                if company in text:
                    tickers.append(ticker)
        
    # Return unique tickers only
    return list(set(tickers))

# 4: Collect posts from WSB
def collect_wsb_posts(limit=100):
    print(f"Collecting {limit} posts from r/WallStreetBets...")

    subreddit = reddit.subreddit("wallstreetbets")

    posts_data = []

    counter = 0
    post_counter = 0
    # Get recent posts
    for post in subreddit.new(limit=limit):
        post_counter += 1
        if post_counter % 500 == 0:
            print(f"Processed {post_counter} posts...")
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
            counter += 1
            if counter % 100 == 0:
                print(f"\tCollected {counter} posts with ticker mentions...")

    # Convert to DataFrame
    df = pd.DataFrame(posts_data)
    print(f"Collected {len(df)} posts with ticker mentions")

    return df

# 5: Run the collection
if __name__ == "__main__":
    
    posts_to_collect = 10000 # Reddit PRAW limits to ~ 10,000 most recent posts
    
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
    
    