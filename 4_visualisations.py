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