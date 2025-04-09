import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For formatting y-axis as percentage

# --- !! Configuration: Set to False to load real data !! ---
USE_FAKE_DATA = True
# --- !! End Configuration !! ---


# --- 1. Load and Prepare Data ---

if USE_FAKE_DATA:
    print("--- Using Fake Data for Testing ---")
    # Generate fake daily FX data (trading days only)
    num_days = 3 * 252 # Approx 3 years of trading days
    # Use business day frequency which excludes weekends
    fx_dates = pd.bdate_range(start='2021-01-01', periods=num_days, freq='B')

    fx_tickers = ['FX_A', 'FX_B', 'FX_C']
    fx_data = {}
    for ticker in fx_tickers:
        # Simulate a simple random walk for prices (starting at 100)
        # Generate daily log returns ~ N(0.0001, 0.01^2)
        log_returns = np.random.normal(loc=0.0001, scale=0.01, size=num_days)
        # Cumulatively sum log returns and add log(100)
        log_prices = np.log(100) + np.cumsum(log_returns)
        # Convert back to prices
        prices = np.exp(log_prices)
        fx_data[ticker] = prices

    fx_df = pd.DataFrame(fx_data, index=fx_dates)
    # Introduce some random missing holidays (NaNs) for specific currencies
    for ticker in fx_tickers:
      random_holidays = fx_df.sample(frac=0.02).index # ~2% random holidays
      fx_df.loc[random_holidays, ticker] = np.nan

    # Forward fill NaNs using the recommended method (addressing FutureWarning)
    fx_df.ffill(inplace=True)
    # Ensure no leading NaNs remain after ffill
    fx_df.dropna(inplace=True)


    # Generate fake monthly strategy data (indexed by last trading day)
    strategy_ticker = 'FakeStrategy'
    # Get month-end dates within the FX date range (using 'ME' for FutureWarning)
    monthly_dates_calendar = pd.date_range(start=fx_df.index.min(), end=fx_df.index.max(), freq='ME')
    # Find the closest preceding *trading day* available in our fx_dates index for each calendar month end
    strategy_dates = fx_df.index.searchsorted(monthly_dates_calendar, side='right') - 1
    # Ensure indices are valid and unique
    valid_indices = strategy_dates[strategy_dates >= 0]
    unique_valid_indices = np.unique(valid_indices)
    strategy_index_dates = fx_df.index[unique_valid_indices]

    # Simulate strategy index value (simple random walk starting at 1000)
    num_months = len(strategy_index_dates)
    monthly_log_returns = np.random.normal(loc=0.005, scale=0.03, size=num_months) # e.g. 0.5% avg monthly return, 3% vol
    strat_log_values = np.log(1000) + np.cumsum(monthly_log_returns)
    strat_values = np.exp(strat_log_values)

    strat_df = pd.DataFrame({strategy_ticker: strat_values}, index=strategy_index_dates)
    strat_df.index.name = 'Date'

    print(f"Generated fake FX data ({fx_df.shape[0]} days, {len(fx_tickers)} tickers)")
    print(f"Generated fake Strategy data ({strat_df.shape[0]} months)")
    print("--- End Fake Data Generation ---")

else:
    print("--- Loading Real Data from Files ---")
    # Load your monthly strategy data (indexed by last trading day)
    strategy_file = 'your_strategy_monthly_data.csv' # <-- PUT YOUR REAL FILENAME HERE
    strategy_ticker = 'YourStrategyTicker' # <-- PUT YOUR REAL TICKER HERE

    strat_df = pd.read_csv(strategy_file, parse_dates=['Date'], index_col='Date')
    strat_df.sort_index(inplace=True) # Ensure data is sorted by date

    # Load your daily FX data (trading days only)
    fx_file = 'your_fx_daily_data.csv' # <-- PUT YOUR REAL FILENAME HERE

    fx_df = pd.read_csv(fx_file, parse_dates=['Date'], index_col='Date')
    fx_df.sort_index(inplace=True) # Ensure data is sorted by date

    # Identify the FX currency columns
    fx_tickers = fx_df.columns.tolist()
    print(f"Using FX tickers for volatility calculation: {fx_tickers}")
    print("--- End Real Data Loading ---")


# --- The rest of the analysis code remains the same ---

# --- 2. Calculate Daily FX Log Returns ---
# Ensure calculation only happens on numeric FX columns
fx_numeric_cols = fx_df.select_dtypes(include=np.number).columns
if not all(ticker in fx_numeric_cols for ticker in fx_tickers):
     print("Warning: Some specified fx_tickers are not numeric columns!")
fx_log_returns = np.log(fx_df[fx_tickers] / fx_df[fx_tickers].shift(1))

# --- 3. Calculate 63-Trading-Day Realized Volatility (Annualized) ---
rolling_window = 63
annualization_factor = np.sqrt(252)
fx_realized_vol = fx_log_returns.rolling(window=rolling_window, min_periods=int(rolling_window*0.8)).std() * annualization_factor # Added min_periods

# --- 4. Calculate Average Daily Realized Volatility ---
average_daily_vol = fx_realized_vol[fx_tickers].mean(axis=1)
average_daily_vol = average_daily_vol.dropna()

# --- 5. Calculate Monthly Strategy Returns (Last Trading Day to Last Trading Day) ---
monthly_strategy_returns = strat_df[strategy_ticker].pct_change()
monthly_strategy_returns.name = 'StrategyReturn'
monthly_strategy_returns = monthly_strategy_returns.to_frame()

# === FIX: Shift the index *values* instead of using index.shift() ===
# This gets the previous date from the index sequence, regardless of frequency
monthly_strategy_returns['PeriodStartDate'] = monthly_strategy_returns.index.to_series().shift(1)
# ====================================================================

monthly_strategy_returns = monthly_strategy_returns.dropna() # Drop first row with NaN return and NaN StartDate

# --- 6. Calculate Average Volatility *During* Each Strategy Return Period ---
def get_average_vol_for_period(period_end_date, period_start_date, daily_vol_series):
    # Ensure start date is valid before filtering
    if pd.isna(period_start_date):
        return np.nan
    mask = (daily_vol_series.index > period_start_date) & (daily_vol_series.index <= period_end_date)
    period_vol = daily_vol_series[mask]
    if period_vol.empty:
        return np.nan
    else:
        return period_vol.mean()

monthly_strategy_returns['AvgVolDuringPeriod'] = monthly_strategy_returns.apply(
    lambda row: get_average_vol_for_period(row.name, row['PeriodStartDate'], average_daily_vol),
    axis=1
)
analysis_df = monthly_strategy_returns.dropna(subset=['AvgVolDuringPeriod'])

if analysis_df.shape[0] < 10:
     print("Warning: Very few periods remain after aligning volatility with strategy returns.")

# --- 7. Create Volatility Quintiles ---
quintile_labels = ['Q1 (Low Vol)', 'Q2', 'Q3', 'Q4', 'Q5 (High Vol)']
try:
    # Ensure there are enough data points for quintiles
    if analysis_df.shape[0] >= 5:
         analysis_df['VolQuintile'] = pd.qcut(analysis_df['AvgVolDuringPeriod'], q=5, labels=False, duplicates='drop')
         # Map numerical quintiles (0-4) to descriptive labels only if qcut succeeded
         analysis_df['VolQuintile'] = analysis_df['VolQuintile'].map(dict(enumerate(quintile_labels)))
    else:
         print("Not enough data points to create 5 quintiles.")
         quintile_labels = [] # Prevent reindexing error later
except ValueError as e:
    print(f"Error creating quintiles: {e}. Check if AvgVolDuringPeriod has enough unique values.")
    quintile_labels = [] # Prevent reindexing error later if labels failed


# --- 8. Calculate Average Strategy Return per Quintile ---
# Check if 'VolQuintile' exists and has data before grouping
if 'VolQuintile' in analysis_df.columns and not analysis_df['VolQuintile'].empty:
    quintile_returns = analysis_df.groupby('VolQuintile')['StrategyReturn'].mean()
    # Ensure the quintiles are ordered correctly for plotting (Q1 to Q5) if labels were assigned
    if quintile_labels: # Check if labels were successfully created and assigned
         try:
            # Only reindex if quintile_returns is not empty
            if not quintile_returns.empty:
                 quintile_returns = quintile_returns.reindex(quintile_labels)
            else:
                 print("Warning: No data found for any quintile.")
         except KeyError:
             print("Warning: Not all quintile labels found in grouped data. Plot might be incomplete.")
             # This can happen if some quintiles had no data after filtering/dropping NaNs
    print("\nAverage Strategy Return per Volatility Quintile (Aligned Periods):")
    print(quintile_returns)

    # --- 9. Visualize ---
    # Only plot if there's something to plot
    if not quintile_returns.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        quintile_returns.plot(kind='bar', ax=ax, color='mediumseagreen', edgecolor='black') # Changed color
        ax.set_title(f'Average Return of {strategy_ticker} by FX Volatility Quintile (Aligned Periods)')
        ax.set_xlabel('Average Realized Volatility Quintile (63-Trading Day FX Vol during period)')
        ax.set_ylabel('Average Strategy Return (Last Trading Day to Last Trading Day)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("\nSkipping plot as no valid quintile return data was calculated.")

else:
    print("\nSkipping grouping and plotting as Volatility Quintiles could not be created or were empty.")

