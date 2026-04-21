# Follow-Up Strategy Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `notebooks/main_analysis.ipynb` with three follow-up strategies — strong-only, strong+high-vol, strong+RSI-confirmed — and produce a master comparison table with interaction heatmaps.

**Architecture:** All new logic is appended as cells to the existing notebook. New strategy signal generators follow the existing pattern (return a numpy array with values in `{-1, 0, 1}`). A new shared `evaluate_signals` function handles all per-strategy metric computation so no evaluation logic is duplicated. RSI is added as a feature-engineering function alongside the existing `compute_zscore`.

**Tech Stack:** Python 3, pandas, numpy, arch, matplotlib, seaborn, Jupyter (existing stack — no new packages needed)

---

## File Structure

| File | Change |
|------|--------|
| `notebooks/main_analysis.ipynb` | **Modify** — append new cells after the existing `save-tables` cell |
| `outputs/tables/strategy_comparison.csv` | **Create** — master comparison table |
| `outputs/tables/interaction_heatmap_data.csv` | **Create** — regime × signal-strength pivot data |
| `outputs/figures/plot6_strategy_comparison.png` | **Create** — bar chart of mean returns by strategy and horizon |
| `outputs/figures/plot7_interaction_heatmap.png` | **Create** — 3-panel heatmap (one per horizon) |
| `outputs/figures/plot8_signal_count_tradeoff.png` | **Create** — signal count vs mean return scatter/bar chart |

---

## Task 1: Add `evaluate_signals` Shared Function

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after existing function cells (before `results-spy-primary` cell)

The existing `evaluate_strategy` groups by a column, which doesn't suit per-strategy comparison. This new function takes a pre-filtered signal column and returns metrics for each horizon.

- [ ] **Step 1: Add cell with `evaluate_signals` function**

Insert a new cell immediately after the `evaluation-function` cell (which defines `evaluate_strategy`). The new cell content:

```python
def evaluate_signals(df, signal_col, horizons=None):
    """Evaluate a named signal column against forward returns for all horizons.
    
    Returns a DataFrame with columns:
    horizon, count, mean_return, median_return, std_return, hit_rate, sharpe_like
    """
    if horizons is None:
        horizons = HORIZONS
    fwd_cols = [f'fwd_return_{h}d' for h in horizons]
    active = df[df[signal_col] != 0].dropna(subset=fwd_cols + [signal_col]).copy()
    rows = []
    for h in horizons:
        fwd_col = f'fwd_return_{h}d'
        strat_ret = active[signal_col] * active[fwd_col]
        std = strat_ret.std()
        rows.append({
            'horizon': f'{h}d',
            'count': int(strat_ret.count()),
            'mean_return': strat_ret.mean(),
            'median_return': strat_ret.median(),
            'std_return': std,
            'hit_rate': (strat_ret > 0).mean(),
            'sharpe_like': strat_ret.mean() / std if std > 0 else np.nan,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Verify function is callable**

Add a quick smoke-test cell after it:

```python
# Quick smoke test — baseline signal column
spy_df_temp = all_results['SPY'][0].copy()
spy_df_temp['baseline_signal'] = spy_df_temp['signal']
_test = evaluate_signals(spy_df_temp, 'baseline_signal')
assert list(_test['horizon']) == ['1d', '3d', '5d'], "horizon column wrong"
assert (_test['count'] > 0).all(), "no signals found"
print("evaluate_signals smoke test PASSED")
print(_test.to_string(index=False))
del spy_df_temp, _test
```

Expected output: prints `evaluate_signals smoke test PASSED` and a 3-row table with positive counts.

---

## Task 2: Add Strategy Signal Generators

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after `evaluate_signals` cell

- [ ] **Step 1: Add cell with all three strategy signal generators**

```python
STRONG_THRESHOLD = 2.0
RSI_WINDOW = 14

def compute_rsi(series, window=RSI_WINDOW):
    """Standard RSI using Wilder's smoothed moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_strong_only_signals(df):
    """Long when zscore <= -2, short when zscore >= 2."""
    return np.where(df['zscore'] <= -STRONG_THRESHOLD, 1,
           np.where(df['zscore'] >= STRONG_THRESHOLD, -1, 0)).astype(int)


def generate_strong_high_vol_signals(df):
    """Strong signals only during high-volatility regimes."""
    strong = (df['zscore'].abs() >= STRONG_THRESHOLD) & (df['vol_regime'] == 'high')
    direction = np.where(df['zscore'] <= -STRONG_THRESHOLD, 1,
                np.where(df['zscore'] >= STRONG_THRESHOLD, -1, 0))
    return np.where(strong, direction, 0).astype(int)


def generate_strong_rsi_confirmed_signals(df):
    """Strong signals confirmed by RSI momentum exhaustion."""
    long_mask  = (df['zscore'] <= -STRONG_THRESHOLD) & (df['rsi'] < 30)
    short_mask = (df['zscore'] >=  STRONG_THRESHOLD) & (df['rsi'] > 70)
    return np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(int)
```

- [ ] **Step 2: Verify generators produce correct output shapes**

Add a smoke-test cell immediately after:

```python
_spy = all_results['SPY'][0].copy()
_spy['rsi'] = compute_rsi(_spy['adj_close'])

_so  = generate_strong_only_signals(_spy)
_shv = generate_strong_high_vol_signals(_spy)
_src = generate_strong_rsi_confirmed_signals(_spy)

assert set(np.unique(_so))  <= {-1, 0, 1}, "strong_only: unexpected values"
assert set(np.unique(_shv)) <= {-1, 0, 1}, "strong_high_vol: unexpected values"
assert set(np.unique(_src)) <= {-1, 0, 1}, "strong_rsi_confirmed: unexpected values"

# strong+high_vol should have fewer signals than strong_only
assert (_shv != 0).sum() <= (_so != 0).sum(), "strong_high_vol should be subset of strong_only"
# rsi confirmed should have fewer signals than strong_only
assert (_src != 0).sum() <= (_so != 0).sum(), "rsi_confirmed should be subset of strong_only"

print("Signal generator smoke tests PASSED")
print(f"  strong_only signals:        {(_so != 0).sum()}")
print(f"  strong_high_vol signals:    {(_shv != 0).sum()}")
print(f"  strong_rsi_confirmed signals: {(_src != 0).sum()}")
del _spy, _so, _shv, _src
```

Expected output: all assertions pass, signal counts decrease progressively (strong_only > strong_high_vol, strong_only > rsi_confirmed).

---

## Task 3: Run All Strategies on SPY

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after smoke-test cells, as the primary results section

- [ ] **Step 1: Add section header cell (markdown)**

```markdown
## Follow-Up Strategy Analysis

Comparing four strategies on SPY using the same GARCH-based pipeline:
- **baseline**: all signals with |z| ≥ 1
- **strong_only**: |z| ≥ 2
- **strong_high_vol**: |z| ≥ 2 AND high-volatility regime
- **strong_rsi_confirmed**: |z| ≥ 2 AND RSI < 30 (long) or RSI > 70 (short)
```

- [ ] **Step 2: Add cell to build per-strategy DataFrames for SPY**

```python
spy_df = all_results['SPY'][0].copy()
spy_df['rsi'] = compute_rsi(spy_df['adj_close'])

spy_df['sig_baseline']          = spy_df['signal']  # existing signal (|z| >= 1)
spy_df['sig_strong_only']       = generate_strong_only_signals(spy_df)
spy_df['sig_strong_high_vol']   = generate_strong_high_vol_signals(spy_df)
spy_df['sig_strong_rsi']        = generate_strong_rsi_confirmed_signals(spy_df)
```

- [ ] **Step 3: Add cell to evaluate all four strategies and print comparison**

```python
strategy_names = {
    'sig_baseline':        'baseline',
    'sig_strong_only':     'strong_only',
    'sig_strong_high_vol': 'strong_high_vol',
    'sig_strong_rsi':      'strong_rsi_confirmed',
}

spy_strategy_results = {}
for col, name in strategy_names.items():
    spy_strategy_results[name] = evaluate_signals(spy_df, col)

# Build master comparison table (wide format)
master_rows = []
for name, result_df in spy_strategy_results.items():
    row = {'strategy': name}
    for _, r in result_df.iterrows():
        h = r['horizon']
        row[f'count_{h}']      = int(r['count'])
        row[f'mean_{h}']       = r['mean_return']
        row[f'hit_rate_{h}']   = r['hit_rate']
        row[f'sharpe_{h}']     = r['sharpe_like']
    master_rows.append(row)

master_table = pd.DataFrame(master_rows).set_index('strategy')
cols_ordered = [f'{m}_{h}' for h in ['1d', '3d', '5d']
                             for m in ['count', 'mean', 'hit_rate', 'sharpe']]
master_table = master_table[cols_ordered]

print("=== Master Strategy Comparison (SPY) ===")
print(master_table.to_string())
```

Expected output: 4-row table with all strategies, counts decrease from baseline → strong_only → filtered variants.

- [ ] **Step 4: Save master table to CSV**

```python
master_table.to_csv(os.path.join(TABLES_DIR, 'strategy_comparison.csv'))
print(f"Saved: {os.path.join(TABLES_DIR, 'strategy_comparison.csv')}")
```

---

## Task 4: Build Interaction Heatmap (Regime × Signal-Strength)

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after Task 3 cells

- [ ] **Step 1: Add cell computing interaction pivot tables**

```python
# Compute strategy return for every row (using the baseline signal direction)
# Interaction is vol_regime x signal_strength — same as existing C3/C5 analysis
# but we report all three forward return horizons side-by-side

interaction_rows = []
active_spy = spy_df[(spy_df['sig_baseline'] != 0)].dropna(
    subset=['vol_regime', 'signal_strength', 'fwd_return_1d', 'fwd_return_3d', 'fwd_return_5d']
).copy()

for h in HORIZONS:
    fwd_col = f'fwd_return_{h}d'
    active_spy[f'strat_ret_{h}d'] = active_spy['sig_baseline'] * active_spy[fwd_col]

interaction_mean = {}
interaction_hitrate = {}
for h in HORIZONS:
    interaction_mean[h] = active_spy.pivot_table(
        index='vol_regime',
        columns='signal_strength',
        values=f'strat_ret_{h}d',
        aggfunc='mean',
        observed=True,
    ).reindex(index=['low', 'medium', 'high'], columns=['weak', 'medium', 'strong'])

    interaction_hitrate[h] = active_spy.pivot_table(
        index='vol_regime',
        columns='signal_strength',
        values=f'strat_ret_{h}d',
        aggfunc=lambda x: (x > 0).mean(),
        observed=True,
    ).reindex(index=['low', 'medium', 'high'], columns=['weak', 'medium', 'strong'])

print("Interaction tables (mean return):")
for h in HORIZONS:
    print(f"\n  Horizon {h}d:")
    print(interaction_mean[h].to_string())
```

- [ ] **Step 2: Save interaction data**

```python
# Save mean-return pivot for each horizon as one stacked CSV
interaction_export = []
for h in HORIZONS:
    df_h = interaction_mean[h].copy()
    df_h.index.name = 'vol_regime'
    df_h.columns.name = 'signal_strength'
    df_h['horizon'] = f'{h}d'
    interaction_export.append(df_h.reset_index().melt(
        id_vars=['vol_regime', 'horizon'], var_name='signal_strength', value_name='mean_return'
    ))

interaction_csv = pd.concat(interaction_export, ignore_index=True)
interaction_csv.to_csv(os.path.join(TABLES_DIR, 'interaction_heatmap_data.csv'), index=False)
print(f"Saved: {os.path.join(TABLES_DIR, 'interaction_heatmap_data.csv')}")
```

---

## Task 5: Generate All Visualizations

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after Task 4 cells

- [ ] **Step 1: Add Plot 6 — Strategy mean return comparison bar chart**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
strategy_colors = {
    'baseline':            '#4c78a8',
    'strong_only':         '#f58518',
    'strong_high_vol':     '#54a24b',
    'strong_rsi_confirmed':'#e45756',
}

for ax, h in zip(axes, ['1d', '3d', '5d']):
    col = f'mean_{h}'
    values = master_table[col]
    bars = ax.bar(values.index, values.values,
                  color=[strategy_colors[s] for s in values.index])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f'Mean Strategy Return — {h}')
    ax.set_ylabel('Mean Log Return')
    ax.set_xticklabels(values.index, rotation=20, ha='right', fontsize=9)

fig.suptitle('Strategy Comparison: Mean Returns by Horizon (SPY)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'plot6_strategy_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot6_strategy_comparison.png")
```

- [ ] **Step 2: Add Plot 7 — Interaction heatmap (regime × signal strength, 3 horizons)**

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, h in zip(axes, HORIZONS):
    sns.heatmap(
        interaction_mean[h],
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        center=0,
        ax=ax,
        cbar=True,
    )
    ax.set_title(f'Horizon: {h}d')
    ax.set_xlabel('Signal Strength')
    ax.set_ylabel('Volatility Regime')

fig.suptitle('Mean Strategy Return: Regime × Signal Strength (SPY)', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'plot7_interaction_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot7_interaction_heatmap.png")
```

- [ ] **Step 3: Add Plot 8 — Signal count vs mean 5d return tradeoff**

```python
fig, ax = plt.subplots(figsize=(8, 5))

for name in strategy_names.values():
    count = master_table.loc[name, 'count_5d']
    mean  = master_table.loc[name, 'mean_5d']
    ax.scatter(count, mean, s=120, color=strategy_colors[name], zorder=5, label=name)
    ax.annotate(name, (count, mean),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_xlabel('Number of Signals (5d horizon)')
ax.set_ylabel('Mean Strategy Return (5d)')
ax.set_title('Signal Count vs Mean Return Tradeoff (SPY, 5d)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'plot8_signal_count_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot8_signal_count_tradeoff.png")
```

---

## Task 6: Cross-Asset Extension

**Files:**
- Modify: `notebooks/main_analysis.ipynb` — append after Task 5 cells

- [ ] **Step 1: Add cell running all four strategies across all tickers**

```python
print("=== Cross-Asset Follow-Up Strategy Analysis ===")

cross_asset_rows = []

for ticker in TICKERS:
    if all_results[ticker] is None:
        print(f"Skipping {ticker} (pipeline failed earlier)")
        continue

    df_t = all_results[ticker][0].copy()
    df_t['rsi'] = compute_rsi(df_t['adj_close'])

    df_t['sig_baseline']        = df_t['signal']
    df_t['sig_strong_only']     = generate_strong_only_signals(df_t)
    df_t['sig_strong_high_vol'] = generate_strong_high_vol_signals(df_t)
    df_t['sig_strong_rsi']      = generate_strong_rsi_confirmed_signals(df_t)

    for col, name in strategy_names.items():
        res = evaluate_signals(df_t, col)
        row = {'ticker': ticker, 'strategy': name}
        for _, r in res.iterrows():
            h = r['horizon']
            row[f'count_{h}']    = int(r['count'])
            row[f'mean_{h}']     = r['mean_return']
            row[f'hit_rate_{h}'] = r['hit_rate']
        cross_asset_rows.append(row)

cross_asset_df = pd.DataFrame(cross_asset_rows)
print(cross_asset_df.to_string(index=False))
```

- [ ] **Step 2: Save cross-asset results**

```python
cross_asset_df.to_csv(os.path.join(TABLES_DIR, 'cross_asset_strategy_comparison.csv'), index=False)
print(f"Saved: {os.path.join(TABLES_DIR, 'cross_asset_strategy_comparison.csv')}")
```

- [ ] **Step 3: Print final summary of new output files**

```python
print("\n=== New Output Files ===")
new_tables = ['strategy_comparison.csv', 'interaction_heatmap_data.csv', 'cross_asset_strategy_comparison.csv']
new_figures = ['plot6_strategy_comparison.png', 'plot7_interaction_heatmap.png', 'plot8_signal_count_tradeoff.png']
for f in new_tables:
    path = os.path.join(TABLES_DIR, f)
    exists = os.path.exists(path)
    print(f"  [{'OK' if exists else 'MISSING'}] tables/{f}")
for f in new_figures:
    path = os.path.join(FIGURES_DIR, f)
    exists = os.path.exists(path)
    print(f"  [{'OK' if exists else 'MISSING'}] figures/{f}")
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| `strong_only` strategy | Task 2 + Task 3 |
| `strong_high_vol` strategy | Task 2 + Task 3 |
| `compute_rsi` function | Task 2 |
| `strong_rsi_confirmed` strategy | Task 2 + Task 3 |
| Shared `evaluate_signals` function | Task 1 |
| Master comparison table (all 4 strategies) | Task 3 |
| Interaction heatmap (regime × strength) | Task 4 + Task 5 |
| Signal count tradeoff chart | Task 5 |
| Strategy comparison bar chart | Task 5 |
| Cross-asset extension | Task 6 |
| Signal counts in all outputs | Every evaluate call includes `count` |
| mean, median, std, hit_rate, sharpe-like | Task 1 (evaluate_signals returns all) |
| RSI window = 14 | Task 2 (`RSI_WINDOW = 14`) |
| Strong threshold = |z| >= 2 | Task 2 (`STRONG_THRESHOLD = 2.0`) |
| High-vol confirmation uses tercile regime | Reuses existing `vol_regime == 'high'` from GARCH pipeline |

### No Placeholder Scan

- No "TBD" or "implement later" present.
- All code blocks are complete.
- All expected outputs described explicitly.
- `strategy_names` dict referenced in Tasks 3, 5, 6 is defined in Task 3 Step 3 — confirmed consistent.
- `interaction_mean` dict defined in Task 4 Step 1, used in Task 5 Step 2 — confirmed consistent.
- `master_table` defined in Task 3 Step 3, used in Task 5 Steps 1 and 3 — confirmed consistent.
- `spy_df` assigned in Task 3 Step 2, used in Tasks 3–5 — confirmed consistent.

### Type Consistency

- `evaluate_signals` returns a DataFrame with columns `horizon, count, mean_return, median_return, std_return, hit_rate, sharpe_like` — all downstream accesses use these exact names.
- All signal generator functions return `np.ndarray` with dtype `int`, values in `{-1, 0, 1}` — consistent with the existing `signal` column.
- `strategy_colors` keys match `strategy_names.values()` exactly: `baseline, strong_only, strong_high_vol, strong_rsi_confirmed`.
