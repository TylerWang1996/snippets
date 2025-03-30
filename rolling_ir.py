import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys # For exiting script if file not found

# --- Configuration ---
ROLLING_WINDOW_MONTHS = 3 * 12  # 3 years
NUM_OUTLIERS_TO_LABEL = 5       # Max number of largest magnitude outliers to label per strategy
FILE_PATH = 'your_total_return_data.csv' # <<< IMPORTANT: Set this to your data file path
DATE_COLUMN_NAME = 'Date'                # <<< IMPORTANT: Set this to the name of your date column

# --- Helper Functions ---

def calculate_rolling_returns(total_return_index_df, window):
    """
    Calculates rolling annualized geometric returns (CAGR) from a
    total return index DataFrame.

    Args:
        total_return_index_df (pd.DataFrame): DataFrame with datetime index and
                                             total return index values per strategy.
        window (int): Rolling window size in months.

    Returns:
        pd.DataFrame: DataFrame containing rolling annualized geometric returns (CAGR).
    """
    # Formula: (End Value / Start Value)^(1 / (Total Periods in Window / Periods Per Year)) - 1
    # Which simplifies to: (End Value / Start Value)^(Periods Per Year / Total Periods in Window) - 1
    rolling_cagr = (total_return_index_df / total_return_index_df.shift(window))**(12.0/window) - 1
    return rolling_cagr.dropna(how='all')

def calculate_rolling_cagr_over_volatility(total_return_index_df, window):
    """
    Calculates rolling ratio of CAGR / Annualized Volatility.
    Numerator: Rolling annualized geometric return (CAGR).
    Denominator: Rolling annualized standard deviation of monthly returns.

    Args:
        total_return_index_df (pd.DataFrame): DataFrame with datetime index and
                                             total return index values per strategy.
        window (int): Rolling window size in months.

    Returns:
        pd.DataFrame: DataFrame containing rolling CAGR / Volatility ratios.
    """
    # Calculate rolling CAGR (Numerator) by calling the dedicated function
    rolling_cagr = calculate_rolling_returns(total_return_index_df, window)

    # Calculate rolling annualized standard deviation of monthly returns (Denominator)
    monthly_returns = total_return_index_df.pct_change() # Keep first NaN for alignment during rolling std calculation
    rolling_annual_std_dev = monthly_returns.rolling(window=window).std() * np.sqrt(12)

    # Calculate CAGR / Volatility Ratio
    # Pandas aligns the DataFrames by index during division
    rolling_ratio = rolling_cagr / rolling_annual_std_dev

    # Replace potential infinities (due to zero std dev) with NaN
    rolling_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any rows where the ratio couldn't be calculated (e.g., std_dev was NaN)
    return rolling_ratio.dropna(how='all')

def plot_rolling_metric_boxplot(data_df, title, ylabel, num_outliers_to_label=5):
    """
    Creates a box-and-whisker plot for rolling metrics and labels outliers.

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
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        if not outliers.empty:
            median = series.median()
            outlier_magnitudes = (outliers - median).abs()
            largest_outliers = outlier_magnitudes.nlargest(num_outliers_to_label)

            for date_index in largest_outliers.index:
                value = outliers.loc[date_index]
                date_str = date_index.strftime('%Y-%m')
                plt.text(i, value, f" {date_str}",
                         horizontalalignment='left', size='small', color='black')

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

# --- Main Analysis ---

# 1. Load Data from File
print(f"--- Loading Data from {FILE_PATH} ---")
try:
    df = pd.read_csv(
        FILE_PATH,
        index_col=DATE_COLUMN_NAME,
        parse_dates=True
    )
    # Or for Excel:
    # df = pd.read_excel(FILE_PATH, index_col=DATE_COLUMN_NAME) #, sheet_name='YourSheetName')

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Index is not DatetimeIndex. Check column '{DATE_COLUMN_NAME}' and parse_dates=True.")
    df = df.sort_index()

    print("Data loaded successfully.")
    print(f"Data Shape: {df.shape}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print("Columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Define Strategies to Analyze (Optional: Filter Columns)
strategies_to_analyze = df.columns.tolist()
# Or specify a subset:
# strategies_to_analyze = ['Your_Strat_1', 'Your_Strat_2']
df_filtered = df[strategies_to_analyze]

# 3. Calculate Rolling Metrics
print(f"\nCalculating {ROLLING_WINDOW_MONTHS}-Month Rolling Annualized Returns (CAGR)...")
rolling_returns_plot = calculate_rolling_returns(df_filtered, ROLLING_WINDOW_MONTHS)

# Calculate the CAGR / Volatility Ratio (labeled as Information Ratio)
# Renamed function for clarity
print(f"\nCalculating {ROLLING_WINDOW_MONTHS}-Month Rolling Information Ratio (CAGR/Volatility)...")
rolling_cagr_vol_ratio = calculate_rolling_cagr_over_volatility(df_filtered, ROLLING_WINDOW_MONTHS)
# Use the desired label "Information Ratio" for display, noting calculation method
metric_name_display = "Information Ratio"
calculation_method = "(CAGR/Vol)" # For clearer labeling on plots

# 4. Plotting
print("\nGenerating plots...")

# Plot Rolling Returns (CAGR)
plot_rolling_metric_boxplot(
    data_df=rolling_returns_plot,
    title=f'{ROLLING_WINDOW_MONTHS//12}-Year Rolling Annualized Returns (CAGR)',
    ylabel='Annualized Return (CAGR)',
    num_outliers_to_label=NUM_OUTLIERS_TO_LABEL
)

# Plot Rolling CAGR / Volatility Ratio (labeled as Information Ratio)
plot_rolling_metric_boxplot(
    data_df=rolling_cagr_vol_ratio,
    title=f'{ROLLING_WINDOW_MONTHS//12}-Year Rolling {metric_name_display} {calculation_method}',
    ylabel=f'{metric_name_display} {calculation_method}', # Label includes calculation method
    num_outliers_to_label=NUM_OUTLIERS_TO_LABEL
)

# Show the plots
plt.show()

print("\nAnalysis Complete.")