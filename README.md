# WSB Sentiment vs Stock Returns

A small end-to-end pipeline that collects recent r/WallStreetBets posts, extracts ticker mentions, scores sentiment, aggregates daily metrics, pulls matching historical prices from Yahoo Finance, and then runs correlation + simple predictive modeling to explore whether WSB chatter relates to short-horizon returns.

## Project structure

```
├── 0_get_wsb_data.py            # Reddit data collection via PRAW
├── 1_preprocess_data.py         # Sentiment analysis and data aggregation
├── 2_get_yfin_data.py           # Stock price data retrieval
├── 3_analysis.py                # Correlation and statistical analysis
├── 4_visualisations.py          # Machine learning models and visualizations
├── yhallsym.txt                 # Stock ticker dictionary
├── wsb_posts.csv                # Raw Reddit posts (generated)
├── wsb_daily_sentiment.csv      # Aggregated daily sentiment (generated)
├── stock_prices.csv             # Historical stock data (generated)
├── merged_data.csv              # Combined dataset (generated)
├── model_results.csv            # Model performance metrics (generated)
└── Final_Report.pdf             # Final report
```

## What it does

1. **Collect WSB posts**

   * Connects to Reddit via PRAW.
   * Loads a ticker universe from `yhallsym.txt`.
   * Extracts tickers from post titles + bodies.
   * Saves a filtered dataset of posts that mention tickers. 

2. **Compute daily sentiment & engagement**

   * Uses TextBlob polarity on each post snippet.
   * Explodes multi-ticker posts.
   * Builds per-day, per-ticker aggregates:

     * mean sentiment
     * total score
     * total comments
     * mention count
     * a simple engagement metric based on score/comments per mention
   * Saves `wsb_daily_sentiment.csv`. 

3. **Pull stock prices**

   * Uses `yfinance` to download OHLCV (uses Close and Volume).
   * Computes same-day return and forward returns for 1/3/7 days.
   * Date range is inferred from the sentiment file with buffers for forward returns.
   * Saves `stock_prices.csv`. 

4. **Merge + correlation analysis**

   * Inner-joins sentiment and price data on `date` and `ticker`.
   * Computes Pearson correlations and p-values for:

     * sentiment vs forward returns
     * mention count vs forward returns
     * engagement vs forward returns
   * Saves `merged_data.csv` and a scatterplot grid `correlation_analysis.png`. 

5. **Simple modeling**

   * Time-series cross-validation.
   * **Linear Regression** to predict forward returns.
   * **Logistic Regression** to predict direction (up/down).
   * Uses standardized features: sentiment, mention_count, engagement.
   * Saves `model_results.csv` and `model_performance.png`. 

## Requirements

* Python 3.9+
* Key libraries:

  * pandas
  * praw
  * textblob
  * yfinance
  * scikit-learn
  * scipy
  * matplotlib

Example install:

```bash
pip install pandas praw textblob yfinance scikit-learn scipy matplotlib
```

(Optional) You may need TextBlob corpora:

```bash
python -m textblob.download_corpora
```

## Setup (Reddit API)

`0_get_wsb_data.py` currently includes Reddit credentials directly in the script. For safety, you should replace these with your own and avoid committing secrets. 

A simple improvement is to use environment variables instead of hardcoding:

```bash
export REDDIT_CLIENT_ID="YOUR_ID"
export REDDIT_CLIENT_SECRET="YOUR_SECRET"
export REDDIT_USER_AGENT="WSB_Sentiment_Analysis_v1.0"
```

Then modify the script to read from `os.environ`.

## Usage

Run scripts in order:

```bash
python 0_get_wsb_data.py
python 1_preprocess_data.py
python 2_get_yfin_data.py
python 3_analysis.py
python 4_visualisations.py
```

### Expected outputs

After a full run you should have:

* `wsb_posts.csv` — raw-ish WSB posts with extracted tickers and a short text snippet.
* `wsb_daily_sentiment.csv` — daily per-ticker sentiment + mention + engagement metrics. 
* `stock_prices.csv` — price, volume, and forward return horizons. 
* `merged_data.csv` — combined dataset used for analysis/modeling. 
* `correlation_analysis.png`
* `model_results.csv` — summary of R² and accuracy by horizon. 
* `model_performance.png`

## Notes & limitations

* **Ticker extraction is heuristic.** It looks for `$TICKER`, uppercase tokens, and even company-name matches from the dictionary; this can create false positives. 
* **Sentiment is lightweight.** TextBlob is a simple baseline and may struggle with WSB sarcasm, memes, and domain-specific slang. 
* **Short-horizon prediction is hard.** Expect modest or noisy correlations and model performance—this project is exploratory, not a trading system.  
* **Survivorship/selection bias.** Only posts that mention tickers and only tickers with available Yahoo data will appear in the merged set.  

## Ideas for extension

* Replace TextBlob with a finance- or social-media-tuned transformer model.
* Add lagged features (yesterday’s sentiment, rolling means/volatility).
* Use robust time-series models (e.g., XGBoost with temporal CV).
* Improve ticker detection with a proper NER + disambiguation step.
* Separate analysis by market regime or volatility buckets.

## License

See `LICENSE`.
