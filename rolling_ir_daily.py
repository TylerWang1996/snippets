import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys # For exiting script if file not found

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252 # Common assumption, adjust if needed
ROLLING_WINDOW_YEARS = 3
ROLLING_WINDOW_DAYS = int(ROLLING_WINDOW_YEARS * TRADING_DAYS_PER_YEAR) # Calculate window in trading days

NUM_OUTLIERS_TO_LABEL = 5      # Max number of largest magnitude outliers to label per strategy (for box plot)
FILE_PATH = 'your_daily_total_return_data.csv' # <<< IMPORTANT: Set this to your DAILY return data file path
DATE_COLUMN_NAME = 'Date'                 # <<< IMPORTANT: Set this to the name of your date column

# --- Helper Functions ---

def calculate_rolling_returns_from_daily(daily_returns_df, window_days):
    """
    Calculates rolling annualized geometric returns (CAGR) from a DataFrame
    of daily total returns. Assumes input dates are trading days.

    Args:
        daily_returns_df (pd.DataFrame): DataFrame with datetime index and
                                         daily total return values (e.g., 0.01 for 1%).
        window_days (int): Rolling window size in trading days.

    Returns:
        pd.DataFrame: DataFrame containing rolling annualized geometric returns (CAGR).
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    # Calculate rolling cumulative return factor using log returns for numerical stability
    # Equivalent to: (1 + daily_returns_df).rolling(window=window_days).apply(np.prod, raw=True)
    log_returns = np.log1p(daily_returns_df) # log(1 + r)
    cumulative_log_returns = log_returns.rolling(window=window_days).sum()
    rolling_total_growth_factor = np.exp(cumulative_log_returns)

    # Annualize the result
    # (Total Growth Factor) ^ (Periods Per Year / Window Length in Periods) - 1
    rolling_cagr = rolling_total_growth_factor**(TRADING_DAYS_PER_YEAR / window_days) - 1

    return rolling_cagr.dropna(how='all')

def calculate_rolling_cagr_over_volatility_from_daily(daily_returns_df, window_days):
    """
    Calculates rolling ratio of CAGR / Annualized Volatility from daily returns.
    Numerator: Rolling annualized geometric return (CAGR).
    Denominator: Rolling annualized standard deviation of daily returns.

    Args:
        daily_returns_df (pd.DataFrame): DataFrame with datetime index and
                                         daily total return values.
        window_days (int): Rolling window size in trading days.

    Returns:
        pd.DataFrame: DataFrame containing rolling CAGR / Volatility ratios.
                     (Often similar to Sharpe Ratio, but using CAGR not arithmetic mean).
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    rolling_cagr = calculate_rolling_returns_from_daily(daily_returns_df, window_days)

    # Calculate rolling annualized standard deviation of *daily* returns
    rolling_daily_std_dev = daily_returns_df.rolling(window=window_days).std()
    rolling_annual_std_dev = rolling_daily_std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)

    rolling_ratio = rolling_cagr / rolling_annual_std_dev
    rolling_ratio.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero std dev

    return rolling_ratio.dropna(how='all')

def plot_rolling_metric_boxplot(data_df, title, ylabel, num_outliers_to_label=5):
    """
    Creates a box-and-whisker plot for rolling metrics and labels outliers.
    (No changes needed in this function itself)

    Args:
        data_df (pd.DataFrame): DataFrame containing the rolling metric values.
        title (str): The title for the plot.
        ylabel (str): The label for the y-axis.
        num_outliers_to_label (int): Max number of outliers to label per strategy.
    """
    if data_df is None or data_df.empty:
        print(f"No data to plot for '{title}'. Skipping plot.")
        return

    plt.figure(figsize=(max(10, len(data_df.columns) * 1.2), 6))
    ax = sns.boxplot(data=data_df, palette="viridis")

    # Identify and label outliers
    for i, strategy in enumerate(data_df.columns):
        series = data_df[strategy].dropna()
        if series.empty: continue

        q1 = series.quantile(0.25); q3 = series.quantile(0.75); iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr; upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        if not outliers.empty:
            median = series.median()
            # Use absolute difference from median for outlier magnitude ranking
            outlier_magnitudes = (outliers - median).abs()
            largest_outliers = outlier_magnitudes.nlargest(num_outliers_to_label)
            for date_index in largest_outliers.index:
                value = outliers.loc[date_index]
                # Format date: Use YYYY-MM-DD for potentially more precise outlier timing
                date_str = date_index.strftime('%Y-%m-%d')
                plt.text(i + 0.02, value, f" {date_str}", horizontalalignment='left', size='small', color='black', verticalalignment='center')


    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_rolling_metric_timeseries(data_df, title, ylabel):
    """
    Creates a time series line plot for rolling metrics for multiple strategies.
    (No changes needed in this function itself)

    Args:
        data_df (pd.DataFrame): DataFrame containing the rolling metric values
                                with datetime index. Columns are strategies.
        title (str): The title for the plot.
        ylabel (str): The label for the y-axis.
    """
    if data_df is None or data_df.empty:
        print(f"No data to plot for '{title}'. Skipping plot.")
        return

    plt.figure(figsize=(12, 6)) # Adjust figure size as needed
    ax = plt.gca() # Get current axes

    for strategy in data_df.columns:
        ax.plot(data_df.index, data_df[strategy], label=strategy, linewidth=1.5) # Slightly thicker line

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Date", fontsize=12)

    # Add legend - adjust position if it overlaps data
    plt.legend(loc='best', fontsize='small')
    # Example for placing legend outside:
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')


    plt.grid(axis='both', linestyle='--', alpha=0.7) # Grid on both axes
    plt.xticks(rotation=0, ha='center') # Adjust rotation if needed
    plt.tight_layout() # Adjust layout (may need adjustment if legend is outside)


# --- Main Analysis ---

# 1. Load Data from File
print(f"--- Loading Daily Return Data from {FILE_PATH} ---")
try:
    # Assuming your CSV has daily returns like 0.01, -0.005 etc.
    df_daily_returns = pd.read_csv(
        FILE_PATH,
        index_col=DATE_COLUMN_NAME,
        parse_dates=True
    )
    # Or for Excel:
    # df_daily_returns = pd.read_excel(FILE_PATH, index_col=DATE_COLUMN_NAME) #, sheet_name='YourSheetName')

    if not isinstance(df_daily_returns.index, pd.DatetimeIndex):
        raise TypeError(f"Index is not DatetimeIndex. Check column '{DATE_COLUMN_NAME}' and parse_dates=True.")

    # Ensure data is sorted by date (important for rolling calculations)
    df_daily_returns = df_daily_returns.sort_index()

    # Optional: Fill potential missing values if needed (e.g., forward fill or fill with 0)
    # df_daily_returns = df_daily_returns.fillna(0) # Example: fill missing daily returns with 0

    print("Daily return data loaded successfully.")
    print(f"Data Shape: {df_daily_returns.shape}")
    print(f"Date Range: {df_daily_returns.index.min()} to {df_daily_returns.index.max()}")
    print("Columns:", df_daily_returns.columns.tolist())
    print(f"Using rolling window: {ROLLING_WINDOW_DAYS} trading days ({ROLLING_WINDOW_YEARS} years)")


except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Define Strategies to Analyze (Optional: Filter Columns)
# By default, use all columns from the loaded daily return data
strategies_to_analyze = df_daily_returns.columns.tolist()
# Or specify a subset:
# strategies_to_analyze = ['Your_Strat_1', 'Your_Strat_2']
df_filtered_daily = df_daily_returns[strategies_to_analyze]

# 3. Calculate Rolling Metrics using Daily Returns
print(f"\nCalculating {ROLLING_WINDOW_YEARS}-Year Rolling Annualized Returns (CAGR) from Daily Data...")
rolling_returns_plot = calculate_rolling_returns_from_daily(df_filtered_daily, ROLLING_WINDOW_DAYS)

print(f"\nCalculating {ROLLING_WINDOW_YEARS}-Year Rolling Return/Volatility Ratio from Daily Data...")
# This ratio is CAGR / Annualized StDev of Daily Returns
rolling_return_vol_ratio = calculate_rolling_cagr_over_volatility_from_daily(df_filtered_daily, ROLLING_WINDOW_DAYS)
metric_name_display = "Return / Volatility Ratio"
calculation_method = "(CAGR / Ann. Daily StDev)" # For clearer labeling on plots

# 4. Plotting
print("\nGenerating plots...")

# Plot 1: Rolling Returns (CAGR) - Box Plot
plot_rolling_metric_boxplot(
    data_df=rolling_returns_plot,
    title=f'{ROLLING_WINDOW_YEARS}-Year Rolling Annualized Returns (CAGR)',
    ylabel='Annualized Return (CAGR)',
    num_outliers_to_label=NUM_OUTLIERS_TO_LABEL
)

# Plot 2: Rolling Return/Volatility Ratio - Box Plot
plot_rolling_metric_boxplot(
    data_df=rolling_return_vol_ratio,
    title=f'{ROLLING_WINDOW_YEARS}-Year Rolling {metric_name_display} {calculation_method} - Distribution',
    ylabel=f'{metric_name_display} {calculation_method}',
    num_outliers_to_label=NUM_OUTLIERS_TO_LABEL
)

# Plot 3: Rolling Return/Volatility Ratio - Time Series Plot
plot_rolling_metric_timeseries(
    data_df=rolling_return_vol_ratio,
    title=f'{ROLLING_WINDOW_YEARS}-Year Rolling {metric_name_display} {calculation_method} - Time Series',
    ylabel=f'{metric_name_display} {calculation_method}'
)

# Show all generated plots
plt.show()

print("\nAnalysis Complete.")