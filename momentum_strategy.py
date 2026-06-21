"""
=============================================================
Quantitative Momentum Strategy Backtest
S&P 500 Stocks | 2013-2018
=============================================================
Description:
    A systematic cross-sectional momentum strategy that:
    - Ranks S&P 500 stocks monthly by 12-1 month momentum
    - Goes long the top 20% (momentum winners)
    - Rebalances monthly
    - Evaluates performance vs buy-and-hold benchmark
    - Adds ML layer to predict momentum signals
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("QUANTITATIVE MOMENTUM STRATEGY BACKTEST")
print("=" * 60)

print("\n[1] Loading data...")
df = pd.read_csv('all_stocks_5yr.csv', parse_dates=['date'])
df = df.sort_values(['Name', 'date']).reset_index(drop=True)
df = df.dropna(subset=['close'])

print(f"    Stocks loaded : {df['Name'].nunique()}")
print(f"    Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"    Total rows    : {len(df):,}")

# ─────────────────────────────────────────────
# 2. COMPUTE MONTHLY PRICES
# ─────────────────────────────────────────────

print("\n[2] Computing monthly prices...")
df['year_month'] = df['date'].dt.to_period('M')

# Use last closing price of each month per stock
monthly = (df.groupby(['Name', 'year_month'])['close']
             .last()
             .unstack(level=0))  # rows = months, columns = stocks

print(f"    Monthly price matrix: {monthly.shape[0]} months × {monthly.shape[1]} stocks")

# ─────────────────────────────────────────────
# 3. COMPUTE MOMENTUM SCORES (12-1 month)
# ─────────────────────────────────────────────

print("\n[3] Computing momentum scores (12-1 month)...")

# 12-month return skipping most recent month
# momentum(t) = price(t-1) / price(t-13) - 1
momentum = monthly.shift(1) / monthly.shift(13) - 1

print(f"    Momentum matrix computed: {momentum.shape}")

# ─────────────────────────────────────────────
# 4. PORTFOLIO CONSTRUCTION — MONTHLY REBALANCE
# ─────────────────────────────────────────────

print("\n[4] Building momentum portfolio (top 20% monthly)...")

# Monthly returns for all stocks
monthly_returns = monthly.pct_change()

portfolio_returns = []
benchmark_returns = []
dates = []
top_stocks_history = []

months = momentum.index[13:]  # skip warmup period

for month in months:
    scores = momentum.loc[month].dropna()
    if len(scores) < 20:
        continue

    # Rank and select top 20% by momentum
    threshold = scores.quantile(0.80)
    winners = scores[scores >= threshold].index.tolist()

    # Equal weight portfolio return next month
    if month not in monthly_returns.index:
        continue

    month_idx = monthly_returns.index.get_loc(month)
    if month_idx + 1 >= len(monthly_returns):
        continue

    next_month = monthly_returns.index[month_idx + 1]
    next_returns = monthly_returns.loc[next_month, winners].dropna()

    if len(next_returns) == 0:
        continue

    port_ret = next_returns.mean()
    bench_ret = monthly_returns.loc[next_month].dropna().mean()

    portfolio_returns.append(port_ret)
    benchmark_returns.append(bench_ret)
    dates.append(next_month.to_timestamp())
    top_stocks_history.append(winners)

# Build results dataframe
results = pd.DataFrame({
    'date': dates,
    'momentum_portfolio': portfolio_returns,
    'benchmark': benchmark_returns
}).set_index('date')

print(f"    Portfolio built over {len(results)} monthly periods")

# ─────────────────────────────────────────────
# 5. PERFORMANCE METRICS
# ─────────────────────────────────────────────

print("\n[5] Computing performance metrics...")

def compute_metrics(returns, label):
    """Compute key quantitative performance metrics."""
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    n_years = len(returns) / 12
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = returns.std() * np.sqrt(12)
    sharpe = annualized_return / annualized_vol
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).mean()

    print(f"\n    {label}:")
    print(f"      Total Return       : {total_return:.1%}")
    print(f"      Annualized Return  : {annualized_return:.1%}")
    print(f"      Annualized Vol     : {annualized_vol:.1%}")
    print(f"      Sharpe Ratio       : {sharpe:.2f}")
    print(f"      Max Drawdown       : {max_drawdown:.1%}")
    print(f"      Win Rate (monthly) : {win_rate:.1%}")

    return {
        'cum_returns': cum_returns,
        'drawdown': drawdown,
        'sharpe': sharpe,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown
    }

port_metrics = compute_metrics(results['momentum_portfolio'], 'Momentum Portfolio (Top 20%)')
bench_metrics = compute_metrics(results['benchmark'], 'Benchmark (Equal Weight S&P 500)')

alpha = port_metrics['annualized_return'] - bench_metrics['annualized_return']
print(f"\n    Alpha (excess return)  : {alpha:.1%}")

# ─────────────────────────────────────────────
# 6. ML LAYER — PREDICT MOMENTUM WINNERS
# ─────────────────────────────────────────────

print("\n[6] ML Layer — predicting momentum winners with Random Forest...")

# Build features for ML
# Features: 1m, 3m, 6m, 12m momentum, volatility
features_list = []

for month in momentum.index[13:]:
    month_idx = monthly_returns.index.get_loc(month) if month in monthly_returns.index else None
    if month_idx is None or month_idx + 1 >= len(monthly_returns):
        continue

    next_month = monthly_returns.index[month_idx + 1]
    stocks = momentum.loc[month].dropna().index

    for stock in stocks:
        try:
            mom_12_1 = momentum.loc[month, stock]
            mom_1 = monthly_returns.loc[month, stock] if stock in monthly_returns.columns else np.nan
            
            # 3 month momentum
            if month_idx >= 3:
                m3 = monthly_returns.index[month_idx - 2]
                mom_3 = (monthly.loc[month, stock] / monthly.loc[m3, stock] - 1) if stock in monthly.columns else np.nan
            else:
                mom_3 = np.nan

            # 6 month momentum
            if month_idx >= 6:
                m6 = monthly_returns.index[month_idx - 5]
                mom_6 = (monthly.loc[month, stock] / monthly.loc[m6, stock] - 1) if stock in monthly.columns else np.nan
            else:
                mom_6 = np.nan

            # Volatility (6 month)
            if month_idx >= 6:
                past_rets = monthly_returns.iloc[month_idx-5:month_idx+1][stock].dropna()
                vol = past_rets.std() if len(past_rets) > 2 else np.nan
            else:
                vol = np.nan

            # Target: is next month return above median?
            next_ret = monthly_returns.loc[next_month, stock] if stock in monthly_returns.columns else np.nan
            median_ret = monthly_returns.loc[next_month].median()
            target = 1 if next_ret > median_ret else 0

            features_list.append({
                'mom_12_1': mom_12_1,
                'mom_1': mom_1,
                'mom_3': mom_3,
                'mom_6': mom_6,
                'volatility': vol,
                'target': target
            })
        except:
            continue

ml_df = pd.DataFrame(features_list).dropna()
print(f"    ML dataset: {len(ml_df):,} stock-month observations")

X = ml_df[['mom_12_1', 'mom_1', 'mom_3', 'mom_6', 'volatility']]
y = ml_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')

print(f"    Random Forest CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# Feature importance
rf.fit(X_scaled, y)
feature_names = ['12-1M Momentum', '1M Return', '3M Momentum', '6M Momentum', 'Volatility']
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(f"\n    Feature Importances:")
for feat, imp in importances.items():
    print(f"      {feat:<20}: {imp:.3f}")

# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

print("\n[7] Generating charts...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Quantitative Momentum Strategy — S&P 500 (2013–2018)',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Chart 1: Cumulative Returns
ax1 = fig.add_subplot(gs[0, :])
(port_metrics['cum_returns'] * 100 - 100).plot(
    ax=ax1, color='#2196F3', linewidth=2.5, label='Momentum Portfolio (Top 20%)')
(bench_metrics['cum_returns'] * 100 - 100).plot(
    ax=ax1, color='#FF9800', linewidth=2, linestyle='--', label='Benchmark (Equal Weight)')
ax1.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax1.fill_between(port_metrics['cum_returns'].index,
                  port_metrics['cum_returns'] * 100 - 100,
                  bench_metrics['cum_returns'] * 100 - 100,
                  alpha=0.15, color='#2196F3')
ax1.set_title('Cumulative Returns (%)', fontweight='bold')
ax1.set_ylabel('Return (%)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Chart 2: Drawdown
ax2 = fig.add_subplot(gs[1, 0])
(port_metrics['drawdown'] * 100).plot(ax=ax2, color='#F44336', linewidth=1.5)
ax2.fill_between(port_metrics['drawdown'].index,
                  port_metrics['drawdown'] * 100, 0, alpha=0.3, color='#F44336')
ax2.set_title('Portfolio Drawdown (%)', fontweight='bold')
ax2.set_ylabel('Drawdown (%)')
ax2.grid(True, alpha=0.3)

# ── Chart 3: Monthly Return Distribution
ax3 = fig.add_subplot(gs[1, 1])
results['momentum_portfolio'].hist(ax=ax3, bins=30, color='#2196F3',
                                    alpha=0.7, edgecolor='white', label='Portfolio')
results['benchmark'].hist(ax=ax3, bins=30, color='#FF9800',
                           alpha=0.5, edgecolor='white', label='Benchmark')
ax3.axvline(0, color='black', linewidth=1)
ax3.set_title('Monthly Return Distribution', fontweight='bold')
ax3.set_xlabel('Monthly Return')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Chart 4: Feature Importance
ax4 = fig.add_subplot(gs[2, 0])
colors = ['#2196F3', '#42A5F5', '#90CAF9', '#BBDEFB', '#E3F2FD']
importances.plot(kind='barh', ax=ax4, color=colors, edgecolor='white')
ax4.set_title('ML Feature Importance\n(Random Forest)', fontweight='bold')
ax4.set_xlabel('Importance Score')
ax4.grid(True, alpha=0.3, axis='x')

# ── Chart 5: Performance Summary Table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
table_data = [
    ['Metric', 'Momentum', 'Benchmark'],
    ['Total Return', f"{port_metrics['total_return']:.1%}", f"{bench_metrics['total_return']:.1%}"],
    ['Ann. Return', f"{port_metrics['annualized_return']:.1%}", f"{bench_metrics['annualized_return']:.1%}"],
    ['Sharpe Ratio', f"{port_metrics['sharpe']:.2f}", f"{bench_metrics['sharpe']:.2f}"],
    ['Max Drawdown', f"{port_metrics['max_drawdown']:.1%}", f"{bench_metrics['max_drawdown']:.1%}"],
    ['Alpha', f"{alpha:.1%}", '—'],
    ['ML Accuracy', f"{cv_scores.mean():.1%}", '—'],
]

table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc='center', loc='center',
                   bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 1:
        cell.set_facecolor('#E3F2FD')
    elif col == 2:
        cell.set_facecolor('#FFF3E0')
    cell.set_edgecolor('white')

ax5.set_title('Performance Summary', fontweight='bold', pad=20)

plt.savefig('momentum_strategy_results.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("    Chart saved: momentum_strategy_results.png")

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
print(f"\n  Momentum Portfolio Alpha : {alpha:.1%} annualized")
print(f"  Sharpe Ratio             : {port_metrics['sharpe']:.2f}")
print(f"  ML Prediction Accuracy   : {cv_scores.mean():.1%}")
print("\n  Files generated:")
print("    momentum_strategy_results.png")
print("\nReady to push to GitHub!")
print("=" * 60)