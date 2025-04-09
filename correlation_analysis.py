import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# --- Constants ---
DEFAULT_ROLLING_YEARS: float = 1.0
DEFAULT_FIXED_LOOKBACK_YEARS: List[int] = [1, 3, 5, 10]
# Default multipliers to convert years to periods based on frequency
DEFAULT_DAILY_FREQ_MULT: int = 252
DEFAULT_MONTHLY_FREQ_MULT: int = 12
# Removed DEFAULT_WEEKLY_FREQ_MULT

# --- Helper Functions ---

def infer_frequency_multiplier(index: pd.DatetimeIndex) -> Optional[int]:
    """
    Infers frequency from DatetimeIndex and returns multiplier for annualization.
    Supports Daily/Business and Monthly frequencies.

    Returns:
        Integer multiplier (e.g., 252 for daily) or None if frequency is unknown/unsupported.
    """
    freq = pd.infer_freq(index)
    if freq:
        freq_str = freq.upper()
        if freq_str.startswith(('D', 'B')): # Daily or Business Day
            print(f"Inferred frequency: Daily/Business (using {DEFAULT_DAILY_FREQ_MULT} periods/year)")
            return DEFAULT_DAILY_FREQ_MULT
        elif freq_str.startswith('M'): # Month End or Start
            print(f"Inferred frequency: Monthly (using {DEFAULT_MONTHLY_FREQ_MULT} periods/year)")
            return DEFAULT_MONTHLY_FREQ_MULT
        # Removed Weekly ('W') check
        # Add more frequencies (e.g., Quarterly 'Q') if needed
    print("Could not infer frequency or frequency is unsupported (Supports Daily/Business, Monthly).")
    return None

def calculate_returns(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame of total return indices to percentage returns.

    Args:
        df_index: DataFrame with DatetimeIndex and asset indices (uses all columns).

    Returns:
        DataFrame of percentage returns. Drops first row with NaNs.
    """
    if not isinstance(df_index.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Basic NaN handling - consider more sophisticated methods if appropriate
    if df_index.isnull().values.any():
        print("Warning: Input DataFrame contains NaNs. Forward-filling before calculating returns.")
        df_index = df_index.ffill()

    # pct_change() operates on all numeric columns by default
    df_returns = df_index.pct_change().dropna()
    print(f"Calculated returns. Result shape: {df_returns.shape}")
    return df_returns

def calculate_rolling_correlations(
    df_returns: pd.DataFrame,
    rolling_period_years: float,
    freq_multiplier: int
) -> Optional[pd.DataFrame]:
    """
    Calculates rolling correlations between all pairs of columns in df_returns.

    Args:
        df_returns: DataFrame of asset returns (uses all columns).
        rolling_period_years: Lookback period in years for the rolling window.
        freq_multiplier: Number of periods per year based on data frequency.

    Returns:
        DataFrame with dates as index and pairwise correlations
        (e.g., 'TickerA_TickerB') as columns, or None if calculation fails.
    """
    window_size = int(rolling_period_years * freq_multiplier)
    if len(df_returns) < window_size:
        print(f"Warning: Not enough data ({len(df_returns)} periods) for rolling window size {window_size}. Skipping rolling correlations.")
        return None

    print(f"Calculating rolling {rolling_period_years}-year correlations (window: {window_size} periods)...")

    # .corr() operates on all numeric columns by default
    rolling_corr = df_returns.rolling(window=window_size).corr(pairwise=True)

    # Check if calculation yielded results (can be all NaN if insufficient overlap)
    if rolling_corr.isnull().all().all() if isinstance(rolling_corr, pd.DataFrame) else rolling_corr.isnull().all():
         print(f"Warning: Rolling correlation calculation resulted in all NaNs (window size: {window_size}). Check data overlap.")
         return None

    # Unstack to get pairs as columns - more Excel friendly
    rolling_corr_unstacked = rolling_corr.unstack(level=1)

    # Clean up column names
    if isinstance(rolling_corr_unstacked.columns, pd.MultiIndex):
         # Assumes structure from .corr().unstack() is (level0=original_col, level1=correlated_col)
         # Example: ('Asset_A', 'Asset_B') -> 'Asset_A_Asset_B'
         # Updated based on testing: .corr() output multiindex is ('level_0_value', 'level_1_value')
         # where level_0 is the first asset and level_1 is the second.
         # After unstack(level=1), columns become MultiIndex like:
         # ('Asset_A', 'Asset_B'), ('Asset_A', 'Asset_C'), ...
         # We need to join these level names.
         rolling_corr_unstacked.columns = [
             f"{col[0]}_{col[1]}" for col in rolling_corr_unstacked.columns.values
         ]


    # Drop self-correlation columns (e.g., 'TickerA_TickerA') using regex
    rolling_corr_unstacked.columns = rolling_corr_unstacked.columns.astype(str)
    self_corr_pattern = r'(.+)_\1'
    rolling_corr_unstacked = rolling_corr_unstacked.loc[:, ~rolling_corr_unstacked.columns.str.match(self_corr_pattern)]

    print("Finished rolling correlations.")
    # Drop rows at the start where the rolling window hasn't filled yet
    return rolling_corr_unstacked.dropna(how='all')


def calculate_fixed_lookback_correlation(
    df_returns: pd.DataFrame,
    years: int,
    freq_multiplier: int
) -> Optional[pd.DataFrame]:
    """
    Calculates the standard correlation matrix for a fixed lookback period
    using all columns in df_returns.

    Args:
        df_returns: DataFrame of asset returns (uses all columns).
        years: Lookback period in years.
        freq_multiplier: Number of periods per year based on data frequency.

    Returns:
        Correlation matrix as a DataFrame, or None if insufficient data.
    """
    lookback_periods = int(years * freq_multiplier)
    if len(df_returns) < lookback_periods:
        print(f"Info: Not enough data ({len(df_returns)} periods) for {years}-year lookback ({lookback_periods} periods). Skipping.")
        return None

    # Select the lookback period from the end of the DataFrame
    df_period = df_returns.iloc[-lookback_periods:]
    print(f"Calculating {years}-year standard correlation ({lookback_periods} periods ending {df_returns.index[-1].date()})...")
    # .corr() operates on all numeric columns by default
    return df_period.corr()


# --- Main Execution Logic ---

def analyze_correlations(
    df_index: pd.DataFrame,
    output_excel_path: str = "correlation_analysis.xlsx",
    rolling_period_years: float = DEFAULT_ROLLING_YEARS,
    fixed_lookback_config: List[int] = DEFAULT_FIXED_LOOKBACK_YEARS
):
    """
    Performs rolling and fixed lookback correlation analysis on ALL asset columns
    in the provided df_index DataFrame and saves results to Excel.

    Args:
        df_index: DataFrame with DatetimeIndex and asset total return indices (uses all columns).
        output_excel_path: Path for the output Excel file.
        rolling_period_years: Lookback period in years for rolling correlations.
        fixed_lookback_config: List of lookback periods (in years) for fixed correlations.
    """
    print("--- Starting Correlation Analysis ---")

    # 1. Prepare Data
    try:
        df_index.index = pd.to_datetime(df_index.index)
    except Exception as e:
        raise ValueError(f"Could not convert DataFrame index to datetime: {e}")

    # calculate_returns uses all columns from df_index
    df_returns = calculate_returns(df_index)
    if df_returns.empty:
        print("Error: Return calculation resulted in an empty DataFrame. Aborting.")
        return

    # Infer frequency or use default (daily)
    freq_mult = infer_frequency_multiplier(df_returns.index)
    if freq_mult is None:
        print(f"Using default frequency multiplier: {DEFAULT_DAILY_FREQ_MULT} (Daily assumption).")
        freq_mult = DEFAULT_DAILY_FREQ_MULT

    # 2. Calculate Rolling Correlations (uses all columns from df_returns)
    rolling_corr_df = calculate_rolling_correlations(
        df_returns, rolling_period_years, freq_mult
    )

    # 3. Calculate Fixed Lookback Correlations (uses all columns from df_returns)
    fixed_correlations: Dict[str, pd.DataFrame] = {}
    for years in fixed_lookback_config:
        corr_matrix = calculate_fixed_lookback_correlation(df_returns, years, freq_mult)
        if corr_matrix is not None:
            fixed_correlations[f"{years}Y Fixed Correlation"] = corr_matrix

    # 4. Export to Excel
    if rolling_corr_df is None and not fixed_correlations:
        print("\nNo correlation results generated to export.")
        return

    print(f"\n--- Exporting results to {output_excel_path} ---")
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # Sheet 1: Rolling Correlations
            if rolling_corr_df is not None and not rolling_corr_df.empty:
                 rolling_corr_df.to_excel(writer, sheet_name='Rolling Correlations', index=True)
                 print(f" - Saved rolling correlations (Sheet: Rolling Correlations)")
            else:
                 print(" - No rolling correlation data to save.")

            # Sheet 2: Fixed Lookback Correlations
            if fixed_correlations:
                current_row = 0
                for name, matrix in fixed_correlations.items():
                    pd.DataFrame([name]).to_excel(
                        writer, sheet_name='Fixed Correlations',
                        startrow=current_row, index=False, header=False
                    )
                    current_row += 1
                    matrix.to_excel(
                        writer, sheet_name='Fixed Correlations',
                        startrow=current_row, index=True
                    )
                    current_row += matrix.shape[0] + 2
                print(f" - Saved fixed lookback correlations (Sheet: Fixed Correlations)")
            else:
                print(" - No fixed correlation data to save.")

        print("--- Excel export complete ---")

    except Exception as e:
        print(f"\nError exporting to Excel: {e}")
        print("Please ensure the file path is valid, the file is not open elsewhere,")
        print("and you have write permissions to the directory.")

# --- Example Usage ---
if __name__ == "__main__":

    # --- Configuration for SAMPLE DATA generation ---
    # NOTE: The analysis functions above work on ALL columns of the DataFrame you provide.
    #       These constants only control the SAMPLE data created below for demonstration.
    NUM_YEARS_SAMPLE_DATA = 6
    NUM_ASSETS_SAMPLE_DATA = 4 # Controls sample data size only
    OUTPUT_FILENAME = "correlation_analysis_final.xlsx"

    # --- Generate Sample Data (Replace with your actual data loading) ---
    print("Generating sample data...")
    end_date = pd.Timestamp('2025-04-08') # Use a fixed recent date
    start_date = end_date - pd.DateOffset(years=NUM_YEARS_SAMPLE_DATA)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Generate tickers for the sample data
    tickers = [f'Asset_{chr(65+i)}' for i in range(NUM_ASSETS_SAMPLE_DATA)]

    # Simulate random walk based indices
    np.random.seed(42)
    price_changes = np.random.randn(len(dates), NUM_ASSETS_SAMPLE_DATA) * 0.01
    data = 100 * (1 + price_changes).cumprod(axis=0)
    sample_df_index = pd.DataFrame(data, index=dates, columns=tickers)

    print("\nSample Data Head:")
    print(sample_df_index.head())
    print("\nSample Data Tail:")
    print(sample_df_index.tail())
    print("-" * 30)

    # --- Run the analysis ---
    # The analyze_correlations function will process ALL columns in sample_df_index
    analyze_correlations(
        df_index=sample_df_index,
        output_excel_path=OUTPUT_FILENAME,
        rolling_period_years=1.0,
        fixed_lookback_config=[1, 3, 5] # Example fixed windows
    )
