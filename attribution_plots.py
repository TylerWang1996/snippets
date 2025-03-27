# -*- coding: utf-8 -*-
"""
Performance Attribution Tool

Calculates and plots single-strategy and difference performance attribution
based on total return index and strategy weight data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime
import sys

# --- Core Calculation Functions ---

def calculate_daily_returns(total_return_index_df: pd.DataFrame,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
    """
    Calculates daily percentage returns from a total return index DataFrame.

    Args:
        total_return_index_df: DataFrame with DateTimeIndex and tickers as columns,
                               containing total return index values.
        start_date: Start date for analysis (YYYY-MM-DD).
        end_date: End date for analysis (YYYY-MM-DD).

    Returns:
        DataFrame with daily percentage returns from start_date to end_date.

    Raises:
        TypeError: If the index cannot be converted to datetime.
        ValueError: If date formats are invalid, start > end, or dates are out of bounds.
        KeyError: If date slicing logic fails.
    """
    if not pd.api.types.is_datetime64_any_dtype(total_return_index_df.index):
        try:
            total_return_index_df.index = pd.to_datetime(total_return_index_df.index)
        except Exception as e:
            raise TypeError(f"Failed to convert DataFrame index to datetime: {e}")

    try:
        start_dt_obj = pd.to_datetime(start_date)
        end_dt_obj = pd.to_datetime(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format for start_date or end_date: {e}")

    if start_dt_obj > end_dt_obj:
        raise ValueError(f"start_date ({start_date}) > end_date ({end_date})")

    df_sorted = total_return_index_df # Assume sorted or sort if needed
    if not df_sorted.index.is_unique:
        print("Warning: Return data index is not unique.")
    if not df_sorted.index.is_monotonic_increasing:
        print("Warning: Return data index is not sorted. Sorting...")
        df_sorted = df_sorted.sort_index()

    # Find date index position for calculation start (need day prior to start_date)
    try:
        calc_start_loc = df_sorted.index.searchsorted(start_dt_obj, side='right')
        if calc_start_loc == 0: # start_date is before the first index date
            calc_start_date = df_sorted.index[0]
            print(f"Warning: Analysis start_date {start_date} is at or before first data point "
                  f"{calc_start_date}. Using first data point.")
        else:
            calc_start_date = df_sorted.index[calc_start_loc - 1]
    except Exception as e:
        raise KeyError(f"Failed to find suitable start date in index near {start_date}: {e}")

    # Find actual end date available in data
    valid_end_dates = df_sorted.index[df_sorted.index <= end_dt_obj]
    if valid_end_dates.empty:
         raise ValueError(f"Analysis end_date '{end_date}' is before the first date in the data.")
    calc_end_date = valid_end_dates[-1]

    # Slice for calculation and compute returns
    returns_data = df_sorted.loc[calc_start_date:calc_end_date].pct_change()

    # Return only the requested date range
    return returns_data.loc[start_date:end_date]


def get_strategy_weights(weights_df: pd.DataFrame,
                         strategy_name: str,
                         start_date: str,
                         end_date: str,
                         target_date_index: pd.DatetimeIndex,
                         date_column_name: str = 'date'
                         ) -> pd.DataFrame:
    """
    Filters, pivots, and aligns weights for one strategy to a target date index.

    Args:
        weights_df: DataFrame in 'long' format with date, ticker, strategy, weight columns.
        strategy_name: The name of the strategy to extract weights for.
        start_date: Start date for the analysis period (YYYY-MM-DD).
        end_date: End date for the analysis period (YYYY-MM-DD).
        target_date_index: The DatetimeIndex (from daily returns) to align weights with.
        date_column_name: Name of the date column in weights_df.

    Returns:
        DataFrame with dates as index, tickers as columns, and weights as values,
        aligned to target_date_index, forward-filled, and NaN filled with 0.

    Raises:
        ValueError: If required columns are missing or pivot fails due to duplicates.
        TypeError: If the date column cannot be converted to datetime.
    """
    required_cols = [date_column_name, 'ticker', 'strategy', 'weight']
    if not all(col in weights_df.columns for col in required_cols):
        raise ValueError(f"Weights data missing columns. Expected: {required_cols}. Found: {weights_df.columns.tolist()}")

    if not pd.api.types.is_datetime64_any_dtype(weights_df[date_column_name]):
        try:
            weights_df[date_column_name] = pd.to_datetime(weights_df[date_column_name])
        except Exception as e:
             raise TypeError(f"Failed to convert weights column '{date_column_name}' to datetime: {e}")

    start_dt_obj = pd.to_datetime(start_date)
    end_dt_obj = pd.to_datetime(end_date)

    # Filter relevant weights (up to end_date for potential ffill, for the specific strategy)
    mask = (weights_df['strategy'] == strategy_name) & (weights_df[date_column_name] <= end_dt_obj)
    filtered_weights = weights_df.loc[mask].copy()

    # Find the last weight entry date <= start_date to ensure correct history for ffill
    possible_start_dates = filtered_weights[filtered_weights[date_column_name] <= start_dt_obj][date_column_name]
    if not possible_start_dates.empty:
        effective_start_date = possible_start_dates.max()
        # Keep only weights from this effective start date onwards
        filtered_weights = filtered_weights[filtered_weights[date_column_name] >= effective_start_date]
    else: # No weights on or before start_date for this strategy in the filtered set
        filtered_weights = pd.DataFrame(columns=weights_df.columns) # Empty DF

    if filtered_weights.empty:
        print(f"Warning: No weights found for strategy '{strategy_name}' relevant to period {start_date}-{end_date}.")
        return pd.DataFrame(0, index=target_date_index, columns=[]) # Return empty structure

    # Pivot from long to wide format (date index, ticker columns)
    try:
        duplicates = filtered_weights[filtered_weights.duplicated([date_column_name, 'ticker'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: Duplicate date/ticker entries found for '{strategy_name}'. Keeping first entry.")
            filtered_weights = filtered_weights.drop_duplicates([date_column_name, 'ticker'], keep='first')

        pivoted_weights = filtered_weights.pivot_table(
            index=date_column_name, columns='ticker', values='weight'
        )
    except Exception as e:
         raise ValueError(f"Failed to pivot weights for '{strategy_name}'. Check for duplicates or data issues. Error: {e}")

    # Align to target index: Reindex to union of dates, forward fill, then select target dates
    pivoted_weights = pivoted_weights.sort_index()
    combined_index = target_date_index.union(pivoted_weights.index)
    aligned_weights = pivoted_weights.reindex(combined_index).ffill()
    aligned_weights = aligned_weights.reindex(target_date_index)

    # Fill any remaining NaNs (e.g., at the start) with 0
    aligned_weights = aligned_weights.fillna(0)

    return aligned_weights


def calculate_single_strategy_attribution(daily_returns: pd.DataFrame,
                                          strategy_weights: pd.DataFrame) -> pd.Series:
    """
    Calculates asset contribution to return for a single strategy (arithmetic).

    Args:
        daily_returns: DataFrame of daily returns (Index=Date, Columns=Tickers).
        strategy_weights: Aligned, wide DataFrame of weights (Index=Date, Columns=Tickers).

    Returns:
        Series with tickers as index and their total contribution over the period (decimal).

    Note: Uses previous day's weights (`shift(1)`) for the current day's return.
          Contribution on the first day will be zero.
    """
    common_tickers = daily_returns.columns.intersection(strategy_weights.columns)
    if common_tickers.empty:
        print("Warning: No common tickers between returns and weights for attribution calc.")
        return pd.Series(dtype=float)

    returns_aligned = daily_returns[common_tickers]
    weights_aligned = strategy_weights[common_tickers]

    # Daily contribution = Beginning-of-Period Weight * Daily Return
    daily_contributions = weights_aligned.shift(1).fillna(0) * returns_aligned

    # Sum daily contributions over the period for each asset
    total_attribution = daily_contributions.sum(skipna=True)

    return total_attribution


def calculate_difference_attribution(daily_returns: pd.DataFrame,
                                     weights_df: pd.DataFrame,
                                     strategy_A_name: str,
                                     strategy_B_name: str,
                                     start_date: str,
                                     end_date: str,
                                     date_column_name: str = 'date'
                                     ) -> pd.Series:
    """
    Calculates the difference in asset contributions between two strategies.

    Args:
        daily_returns: DataFrame of daily returns for relevant assets.
        weights_df: The original *long* DataFrame with weights for all strategies.
        strategy_A_name: Name of the first strategy.
        strategy_B_name: Name of the second strategy (benchmark/comparison).
        start_date: Start date for analysis (YYYY-MM-DD).
        end_date: End date for analysis (YYYY-MM-DD).
        date_column_name: Name of the date column in weights_df.

    Returns:
        Series with tickers as index and the difference in contribution
        (Strategy A Contrib - Strategy B Contrib) over the period (decimal).
    """
    # Get aligned weights for both strategies
    weights_A = get_strategy_weights(weights_df, strategy_A_name, start_date, end_date, daily_returns.index, date_column_name)
    weights_B = get_strategy_weights(weights_df, strategy_B_name, start_date, end_date, daily_returns.index, date_column_name)

    # Identify all tickers present in either strategy's aligned weights
    all_tickers = weights_A.columns.union(weights_B.columns)
    if all_tickers.empty:
         print(f"Warning: No tickers found for '{strategy_A_name}' or '{strategy_B_name}'. Difference attribution empty.")
         return pd.Series(dtype=float)

    # Align A and B weight columns to include all tickers from both, fill missing with 0
    weights_A = weights_A.reindex(columns=all_tickers, fill_value=0)
    weights_B = weights_B.reindex(columns=all_tickers, fill_value=0)

    # Identify tickers present in *both* the combined weights and the returns data
    tickers_in_returns = daily_returns.columns
    relevant_tickers = all_tickers.intersection(tickers_in_returns)
    if relevant_tickers.empty:
        print("Warning: No common tickers between combined weights and returns data. Difference attribution empty.")
        return pd.Series(dtype=float)

    # Align returns and weights to only relevant tickers
    returns_aligned = daily_returns[relevant_tickers]
    weights_A_aligned = weights_A[relevant_tickers]
    weights_B_aligned = weights_B[relevant_tickers]

    # Calculate total contribution for each asset in each strategy
    # Note: This recalculates weights.shift(1) * returns. For performance on huge data,
    # you could refactor to reuse daily contributions if already computed.
    contrib_A = calculate_single_strategy_attribution(returns_aligned, weights_A_aligned)
    contrib_B = calculate_single_strategy_attribution(returns_aligned, weights_B_aligned)

    # Calculate difference, ensuring alignment
    difference = contrib_A.reindex(relevant_tickers).fillna(0) - contrib_B.reindex(relevant_tickers).fillna(0)

    return difference


# --- Plotting Function ---

def plot_attribution(attribution_series: pd.Series,
                     title: str,
                     figsize: tuple = (10, 6),
                     color_positive: str = 'green',
                     color_negative: str = 'red',
                     max_bars: int = None,
                     save_path: str = None) -> plt.Axes:
    """
    Creates and optionally saves a horizontal bar chart for attribution results.

    Args:
        attribution_series: Series with tickers as index and contributions (decimal) as values.
        title: Title for the chart.
        figsize: Figure size tuple.
        color_positive: Color for positive bars.
        color_negative: Color for negative bars.
        max_bars: If set, displays only the top N bars by absolute value.
        save_path: Optional path (including filename) to save the figure.

    Returns:
        Matplotlib Axes object, or None if plotting skipped.
    """
    if attribution_series.empty or attribution_series.isnull().all():
        print(f"Plotting skipped: Input series for '{title}' is empty or all NaN.")
        return None

    plot_data = attribution_series.dropna().copy()

    # Limit bars shown if requested
    if max_bars is not None and len(plot_data) > max_bars:
        top_indices = plot_data.abs().nlargest(max_bars).index
        plot_data = plot_data.loc[top_indices]
        print(f"Info: Displaying top {max_bars} contributors by absolute value for plot '{title}'.")

    if plot_data.empty:
        print(f"Plotting skipped: No data left for '{title}' after filtering.")
        return None

    plot_data = plot_data.sort_values() # Sort smallest to largest for plotting
    colors = [color_positive if x >= 0 else color_negative for x in plot_data.values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(plot_data.index, plot_data.values, color=colors)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))

    ax.set_xlabel("Contribution to Return / Return Difference")
    #ax.set_ylabel("Asset")
    ax.set_title(title)

    # Add value labels to bars
    try:
        ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9,
                     labels=[f'{x*100:.2f}%' for x in plot_data.values])
    except Exception as e:
        print(f"Warning: Could not add bar labels for plot '{title}'. Error: {e}")

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()
    return ax


# --- Data Loading Functions ---

def load_return_data(path: str, index_col: int = 0, parse_dates: bool = True) -> pd.DataFrame:
    """
    Loads total return index data from a CSV file.

    Args:
        path: Path to the return data CSV file.
        index_col: Column index to use as the DataFrame index (usually contains dates).
        parse_dates: Whether to attempt parsing the index column as dates.

    Returns:
        DataFrame with DateTimeIndex and tickers as columns (total return index values).

    Raises:
        FileNotFoundError: If the path is invalid.
        TypeError: If index cannot be converted to datetime.
        Exception: For other pandas read/processing errors.
    """
    print(f"Loading return data from: {path}")
    try:
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError("Loaded return data index is not datetime. Check 'index_col' and 'parse_dates' options.")
        if not df.index.is_unique:
            print("Warning: Loaded return data index is not unique.")
        if not df.index.is_monotonic_increasing:
            print("Warning: Loaded return data index is not sorted. Sorting...")
            df = df.sort_index()

        # Attempt numeric conversion, coercing errors
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.isnull().any().any():
            # Find columns with NaNs introduced by coercion
            nan_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            print(f"Warning: Non-numeric values coerced to NaN in return columns: {nan_cols}. Consider cleaning input data or using fill methods (e.g., ffill).")
            # Example fill: df_numeric = df_numeric.ffill()

        print("Return data loaded successfully.")
        return df_numeric

    except FileNotFoundError:
        print(f"ERROR: Return data file not found at {path}")
        raise
    except Exception as e:
        print(f"ERROR loading or processing return data from {path}: {e}")
        raise


def load_weights_data(path: str, date_column_name: str = 'date') -> pd.DataFrame:
    """
    Loads strategy weights data from a CSV file (long format).

    Args:
        path: Path to the weights data CSV file.
        date_column_name: Name of the column containing dates.

    Returns:
        DataFrame in 'long' format with columns: 'date' (or specified name),
        'ticker', 'strategy', 'weight'.

    Raises:
        FileNotFoundError: If the path is invalid.
        ValueError: If required columns are missing or weight is non-numeric.
        TypeError: If date column cannot be converted to datetime.
        Exception: For other pandas read/processing errors.
    """
    print(f"Loading weights data from: {path}")
    try:
        df = pd.read_csv(path, parse_dates=[date_column_name])

        expected_cols = [date_column_name, 'ticker', 'strategy', 'weight']
        if not all(col in df.columns for col in expected_cols):
             raise ValueError(f"Weights data missing columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")

        if not pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
             # This check might be redundant if parse_dates worked, but good as fallback
             raise TypeError(f"Weights date column '{date_column_name}' not parsed as datetime.")

        if not pd.api.types.is_numeric_dtype(df['weight']):
             print(f"Warning: Weights 'weight' column not numeric. Attempting conversion.")
             df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
             if df['weight'].isnull().any():
                 raise ValueError("Non-numeric values found in 'weight' column after coercion. Clean data required.")

        print("Weights data loaded successfully.")
        return df

    except FileNotFoundError:
        print(f"ERROR: Weights data file not found at {path}")
        raise
    except Exception as e:
        print(f"ERROR loading or processing weights data from {path}: {e}")
        raise


# --- Analysis Execution Function ---

def run_attribution_analysis(total_return_df: pd.DataFrame,
                             weights_df: pd.DataFrame,
                             start_date: str,
                             end_date: str,
                             strategy_A: str,
                             strategy_B: str,
                             weights_date_col: str = 'date',
                             results_suffix: str = None,
                             max_bars_plot: int = 20):
    """
    Runs single (A & B) and difference attribution, prints, and plots results.
    """
    print(f"\n--- Running Attribution Analysis ---")
    print(f"Strategies: '{strategy_A}' vs '{strategy_B}'")
    print(f"Period: {start_date} to {end_date}")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"_{results_suffix}" if results_suffix else ""

    try:
        # --- Calculate Daily Returns ---
        print("\nCalculating daily returns...")
        daily_returns = calculate_daily_returns(total_return_df, start_date, end_date)

        # --- Single Strategy: A ---
        print(f"\n--- Analyzing Strategy: {strategy_A} ---")
        weights_A = get_strategy_weights(weights_df, strategy_A, start_date, end_date, daily_returns.index, weights_date_col)
        attribution_A = calculate_single_strategy_attribution(daily_returns, weights_A)

        if not attribution_A.empty:
            total_return_A = attribution_A.sum()
            print(f"Total Calculated Return ({strategy_A}): {total_return_A:.2%}")
            print(f"Asset Attribution ({strategy_A}):\n{attribution_A.map('{:.2%}'.format)}")
            title_A = f"Performance Attribution: {strategy_A}\n({start_date} to {end_date})"
            save_A = f"attrib_{strategy_A}{suffix}_{timestamp_str}.png"
            plot_attribution(attribution_A, title_A, max_bars=max_bars_plot, save_path=save_A)
        else:
            print(f"Could not calculate attribution for {strategy_A}.")

        # --- Single Strategy: B ---
        print(f"\n--- Analyzing Strategy: {strategy_B} ---")
        weights_B = get_strategy_weights(weights_df, strategy_B, start_date, end_date, daily_returns.index, weights_date_col)
        attribution_B = calculate_single_strategy_attribution(daily_returns, weights_B)

        if not attribution_B.empty:
            total_return_B = attribution_B.sum()
            print(f"Total Calculated Return ({strategy_B}): {total_return_B:.2%}")
            print(f"Asset Attribution ({strategy_B}):\n{attribution_B.map('{:.2%}'.format)}")
            title_B = f"Performance Attribution: {strategy_B}\n({start_date} to {end_date})"
            save_B = f"attrib_{strategy_B}{suffix}_{timestamp_str}.png"
            plot_attribution(attribution_B, title_B, max_bars=max_bars_plot, save_path=save_B)
        else:
            print(f"Could not calculate attribution for {strategy_B}.")

        # --- Difference Attribution ---
        print(f"\n--- Analyzing Difference: {strategy_A} vs {strategy_B} ---")
        difference = calculate_difference_attribution(daily_returns, weights_df, strategy_A, strategy_B, start_date, end_date, weights_date_col)

        if not difference.empty:
            total_difference = difference.sum()
            print(f"Total Calculated Difference ({strategy_A} - {strategy_B}): {total_difference:.2%}")
            print_diff = difference[~np.isclose(difference, 0, atol=1e-6)] # Non-zero differences
            if print_diff.empty:
                print("Difference Attribution by Asset: (No significant differences)")
            else:
                print(f"Difference Attribution by Asset (Non-Zero):\n{print_diff.map('{:.2%}'.format)}")

            title_diff = f"Difference Attribution: {strategy_A} vs {strategy_B}\n({start_date} to {end_date})"
            save_diff = f"diff_{strategy_A}_vs_{strategy_B}{suffix}_{timestamp_str}.png"
            plot_attribution(difference, title_diff, color_positive='darkcyan', color_negative='coral', max_bars=max_bars_plot, save_path=save_diff)
        else:
            print(f"Could not calculate difference attribution between {strategy_A} and {strategy_B}.")

    except (ValueError, TypeError, KeyError, FileNotFoundError) as e:
        print(f"\n--- ANALYSIS FAILED --- \nError: {e}")
        # import traceback # Uncomment for detailed traceback during debugging
        # traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Analysis --- \nError: {e}")
        # import traceback
        # traceback.print_exc()

    print("\nAnalysis complete.")


# --- Simulation Function (for Testing/Example) ---

def generate_simulated_data(sim_start_date_str: str = None,
                           sim_end_date_str: str = None,
                           num_days_history: int = 455) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates sample total return and weights DataFrames for testing/example.

    Args:
        sim_start_date_str: Optional start date (YYYY-MM-DD). Defaults to ~1y 3m before end date.
        sim_end_date_str: Optional end date (YYYY-MM-DD). Defaults to yesterday.
        num_days_history: Number of days history if dates aren't specified.

    Returns:
        Tuple containing (total_return_df, weights_df).
    """
    print("--- Generating Simulated Data for Testing ---")

    # Determine date range
    if sim_end_date_str:
        end_sim_date = pd.to_datetime(sim_end_date_str)
    else:
        end_sim_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

    if sim_start_date_str:
        start_sim_date = pd.to_datetime(sim_start_date_str)
    else:
        start_sim_date = end_sim_date - pd.Timedelta(days=num_days_history - 1)

    dates_tr = pd.date_range(start_sim_date, end_sim_date, freq='D')
    tickers = ['SIM_A', 'SIM_B', 'SIM_C', 'SIM_D', 'SIM_E'] # Simulated tickers

    # Simulate Returns
    np.random.seed(42)
    n_days = len(dates_tr)
    daily_ret_sim = np.random.normal(0.0005, 0.015, size=(n_days, len(tickers))) + 1
    tr_index_values = 100 * np.cumprod(daily_ret_sim, axis=0)
    total_return_df = pd.DataFrame(tr_index_values, index=dates_tr, columns=tickers).sort_index()

    # Simulate Weights (Long Format)
    weights_list = []
    strategies = ['SimGrowth', 'SimValue']
    weight_change_dates = pd.date_range(start_sim_date, end_sim_date, freq='MS') # Monthly changes

    for dt in weight_change_dates:
        # SimGrowth
        w_g = np.random.rand(len(tickers)); w_g /= w_g.sum()
        for i, t in enumerate(tickers):
             if not (t == 'SIM_C' and dt.month % 4 == 0):
                weights_list.append({'date': dt, 'ticker': t, 'strategy': 'SimGrowth', 'weight': w_g[i]})
        # SimValue (different assets)
        val_tickers = ['SIM_A', 'SIM_B', 'SIM_F', 'SIM_G']
        w_v = np.random.rand(len(val_tickers)); w_v /= w_v.sum()
        for i, t in enumerate(val_tickers):
             if not (t == 'SIM_B' and dt.month % 3 == 0):
                 weights_list.append({'date': dt, 'ticker': t, 'strategy': 'SimValue', 'weight': w_v[i]})

    # Add missing tickers from weights to return data if necessary
    all_weight_tickers = set(item['ticker'] for item in weights_list)
    missing_tickers = list(all_weight_tickers - set(total_return_df.columns))
    if missing_tickers:
        print(f"Adding dummy return data for simulated tickers missing from returns: {missing_tickers}")
        for mticker in missing_tickers:
            ret_sim = np.random.normal(0.0003, 0.018, size=(n_days, 1)) + 1
            tr_sim = 100 * np.cumprod(ret_sim, axis=0)
            total_return_df[mticker] = tr_sim.flatten()
        total_return_df = total_return_df.sort_index(axis=1) # Keep columns sorted

    weights_df = pd.DataFrame(weights_list)
    weights_df['date'] = pd.to_datetime(weights_df['date'])

    print(f"Simulated data generated from {start_sim_date.date()} to {end_sim_date.date()}.")
    return total_return_df, weights_df


# --- Main Execution Block ---

if __name__ == "__main__":

    # --- Configuration ---
    # !! MODIFY THESE PATHS and PARAMETERS FOR YOUR DATA !!
    RETURN_DATA_CSV = "path/to/your/total_returns.csv"
    WEIGHTS_DATA_CSV = "path/to/your/strategy_weights.csv"

    # Analysis Period (adjust as needed, defaults use current context)
    default_end_date = pd.Timestamp('2025-03-25')
    default_start_date = default_end_date - pd.Timedelta(days=89) # Approx 3 months prior
    ANALYSIS_START_DATE = default_start_date.strftime('%Y-%m-%d') # e.g., '2024-12-27'
    ANALYSIS_END_DATE = default_end_date.strftime('%Y-%m-%d')     # e.g., '2025-03-25'

    # Strategy Names (MUST match names in weights CSV)
    STRATEGY_A = 'Your_Strategy_Name_A'
    STRATEGY_B = 'Your_Strategy_Name_B' # e.g., a benchmark or another strategy

    # Column name in weights CSV containing dates (if not 'date')
    WEIGHTS_DATE_COLUMN = 'date'

    # Optional suffix for output plot filenames (e.g., "q1_report")
    RESULTS_FILE_SUFFIX = None
    # --- End Configuration ---


    # --- Load Data ---
    print("--- Starting Performance Attribution ---")
    try:
        # Load data using the loading functions
        return_data = load_return_data(RETURN_DATA_CSV, index_col=0, parse_dates=True)
        weights_data = load_weights_data(WEIGHTS_DATA_CSV, date_column_name=WEIGHTS_DATE_COLUMN)
        data_loaded = True
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"\nFATAL ERROR during data loading: {e}")
        print("Please check file paths and data formats.")
        # Optionally: Run with simulated data if loading fails?
        print("\nAttempting to run with SIMULATED data for demonstration...")
        return_data, weights_data = generate_simulated_data()
        STRATEGY_A, STRATEGY_B = 'SimGrowth', 'SimValue' # Use sim names
        data_loaded = True # Assume sim data loaded ok
        #data_loaded = False # Exit if real data fails
    except Exception as e:
        print(f"\nUNEXPECTED FATAL ERROR during data loading: {e}")
        data_loaded = False

    # --- Run Analysis ---
    if data_loaded:
        run_attribution_analysis(
            total_return_df=return_data,
            weights_df=weights_data,
            start_date=ANALYSIS_START_DATE,
            end_date=ANALYSIS_END_DATE,
            strategy_A=STRATEGY_A,
            strategy_B=STRATEGY_B,
            weights_date_col=WEIGHTS_DATE_COLUMN,
            results_suffix=RESULTS_FILE_SUFFIX,
            max_bars_plot=25 # Adjust max bars shown in plots if desired
        )
    else:
        print("\nAnalysis skipped due to data loading errors.")
        sys.exit(1) # Exit with error status

    print("\n--- Performance Attribution Script Finished ---")