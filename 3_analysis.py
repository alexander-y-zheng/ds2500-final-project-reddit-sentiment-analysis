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
