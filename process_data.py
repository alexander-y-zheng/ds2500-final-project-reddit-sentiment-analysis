import pandas as pd
from textblob import TextBlob

# load data
print("Loading Reddit posts...")    
try:    
    df_posts = pd.read_csv('wsb_posts.csv')
    print(f"Loaded {len(df_posts)} posts")
except FileNotFoundError:
    print("Error: 'wsb_posts.csv' file not found.")
    exit(1)


