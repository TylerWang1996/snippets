# -*- coding: utf-8 -*-
"""
Performance Attribution Tool

Calculates and plots single-strategy and difference performance attribution
based on total return index and strategy weight data. Aligns weights effective
from specific dates forward to the daily return dates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime
import sys

# --- Core Calculation Functions ---

# calculate_daily_returns (No changes needed)
def calculate_daily_returns(total_return_index_df: pd.DataFrame,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
    """
    Calculates daily percentage returns from a total return index DataFrame.
    [... full docstring remains the same ...]
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


# --- REVERTED get_strategy_weights Function (Using reindex + ffill) ---
def get_strategy_weights(weights_df: pd.DataFrame,
                         strategy_name: str,
                         start_date: str, # Start date of the *analysis* period
                         end_date: str,   # End date of the *analysis* period
                         target_date_index: pd.DatetimeIndex, # Daily dates from returns
                         date_column_name: str = 'date'
                         ) -> pd.DataFrame:
    """
    Filters, pivots, and aligns strategy weights (effective from specific dates)
    to a target daily date index using forward fill.

    Args:
        weights_df: DataFrame in 'long' format with date, ticker, strategy, weight columns.
                    Dates indicate the effective date for the weights.
        strategy_name: The name of the strategy to extract weights for.
        start_date: Start date for the analysis period (YYYY-MM-DD).
        end_date: End date for the analysis period (YYYY-MM-DD).
        target_date_index: The *daily* DatetimeIndex to align weights with.
        date_column_name: Name of the date column in weights_df.

    Returns:
        DataFrame with the daily target_date_index, tickers as columns,
        weights forward-filled from their effective dates, and NaN filled with 0.

    Raises:
        ValueError: If required columns missing or pivot fails due to duplicates.
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

    # Filter weights for the strategy, up to the end date
    mask = (weights_df['strategy'] == strategy_name) & (weights_df[date_column_name] <= end_dt_obj)
    filtered_weights = weights_df.loc[mask].copy()

    # Find the latest weight entry date AT or BEFORE the analysis start date
    # This ensures we have the correct weights to forward-fill from the beginning
    possible_start_weights = filtered_weights[filtered_weights[date_column_name] <= start_dt_obj]
    if not possible_start_weights.empty:
        effective_start_date = possible_start_weights[date_column_name].max()
        # Keep weights from this effective start date onwards up to analysis end date
        filtered_weights = filtered_weights[filtered_weights[date_column_name] >= effective_start_date]
    else:
        # No weights found at or before the start date. Weights will be 0 until the first entry > start_date.
        # Keep weights after start_date but within end_date
         filtered_weights = filtered_weights[filtered_weights[date_column_name] > start_dt_obj]
         if filtered_weights.empty:
              print(f"Warning: No weights found for strategy '{strategy_name}' relevant to period {start_date}-{end_date}.")
              return pd.DataFrame(0, index=target_date_index, columns=[]) # Return empty structure

    # Pivot from long to wide format (weight effective date index, ticker columns)
    try:
        # Handle potential duplicates for the same effective date/ticker (keep first or last?)
        # Keeping first for now, assuming data shouldn't have duplicates per effective date.
        duplicates = filtered_weights[filtered_weights.duplicated([date_column_name, 'ticker'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: Duplicate date/ticker entries found for '{strategy_name}'. Keeping first entry.")
            filtered_weights = filtered_weights.drop_duplicates([date_column_name, 'ticker'], keep='first')

        pivoted_weights = filtered_weights.pivot_table(
            index=date_column_name, columns='ticker', values='weight'
        )
    except Exception as e:
         raise ValueError(f"Failed to pivot weights for '{strategy_name}'. Check for duplicates or data issues. Error: {e}")

    # Align weights to the target daily index
    # 1. Create a combined index of daily dates and weight effective dates
    # 2. Reindex the pivoted weights to this combined index
    # 3. Forward-fill the weights (carry last known weight forward)
    # 4. Reindex again to keep only the target daily dates

    pivoted_weights = pivoted_weights.sort_index() # Ensure weights are sorted by date
    # Combine the daily target index with the weight effective dates index
    combined_index = target_date_index.union(pivoted_weights.index).sort_values()

    # Reindex pivoted weights to combined index, then forward fill
    aligned_weights_ffilled = pivoted_weights.reindex(combined_index).ffill()

    # Select only the dates present in the original target (daily) index
    aligned_weights = aligned_weights_ffilled.reindex(target_date_index)

    # Fill any remaining NaNs with 0 (e.g., if the first weight date is after the first target date)
    aligned_weights = aligned_weights.fillna(0)

    return aligned_weights


# calculate_single_strategy_attribution (No changes needed)
def calculate_single_strategy_attribution(daily_returns: pd.DataFrame,
                                          strategy_weights: pd.DataFrame) -> pd.Series:
    """
    Calculates asset contribution to return for a single strategy (arithmetic).
    [... full docstring remains the same ...]
    """
    common_tickers = daily_returns.columns.intersection(strategy_weights.columns)
    if common_tickers.empty:
        print("Warning: No common tickers between returns and weights for attribution calc.")
        return pd.Series(dtype=float)

    returns_aligned = daily_returns[common_tickers]
    weights_aligned = strategy_weights[common_tickers]

    # Daily contribution = Beginning-of-Period Weight * Daily Return
    # Weights are now forward-filled daily, shift(1) correctly gets BOP weight
    daily_contributions = weights_aligned.shift(1).fillna(0) * returns_aligned

    # Sum daily contributions over the period for each asset
    total_attribution = daily_contributions.sum(skipna=True)

    return total_attribution


# calculate_difference_attribution (No changes needed)
# Relies on get_strategy_weights, which now uses ffill logic
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
    [... full docstring remains the same ...]
    """
    # Get aligned weights for both strategies using the reverted get_strategy_weights
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
    contrib_A = calculate_single_strategy_attribution(returns_aligned, weights_A_aligned)
    contrib_B = calculate_single_strategy_attribution(returns_aligned, weights_B_aligned)

    # Calculate difference, ensuring alignment
    difference = contrib_A.reindex(relevant_tickers).fillna(0) - contrib_B.reindex(relevant_tickers).fillna(0)

    return difference

# --- Plotting Function --- (No changes needed)
def plot_attribution(attribution_series: pd.Series,
                     title: str,
                     figsize: tuple = (10, 6),
                     color_positive: str = 'green',
                     color_negative: str = 'red',
                     max_bars: int = None,
                     save_path: str = None) -> plt.Axes:
    """
    Creates and optionally saves a horizontal bar chart for attribution results.
    [... full docstring remains the same ...]
    """
    if attribution_series.empty or attribution_series.isnull().all():
        print(f"Plotting skipped: Input series for '{title}' is empty or all NaN.")
        return None

    plot_data = attribution_series.dropna().copy()

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

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    ax.set_xlabel("Contribution to Return / Return Difference")
    ax.set_title(title)

    try:
        ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9,
                     labels=[f'{x*100:.2f}%' for x in plot_data.values])
    except Exception as e:
        print(f"Warning: Could not add bar labels for plot '{title}'. Error: {e}")

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


# --- Data Loading Functions --- (No changes needed)
def load_return_data(path: str, index_col: int = 0, parse_dates: bool = True) -> pd.DataFrame:
    """
    Loads total return index data from a CSV file.
    [... full docstring remains the same ...]
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

        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.isnull().any().any():
            nan_cols = df_numeric.columns[df_numeric.isnull().any()].tolist()
            print(f"Warning: Non-numeric values coerced to NaN in return columns: {nan_cols}. Consider cleaning input data or using fill methods (e.g., ffill).")

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
    [... full docstring remains the same ...]
    """
    print(f"Loading weights data from: {path}")
    try:
        df = pd.read_csv(path, parse_dates=[date_column_name])

        expected_cols = [date_column_name, 'ticker', 'strategy', 'weight']
        if not all(col in df.columns for col in expected_cols):
             raise ValueError(f"Weights data missing columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")

        if not pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
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


# --- Analysis Execution Function --- (No changes needed)
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
    [... full docstring remains the same ...]
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
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Analysis --- \nError: {e}")
        # import traceback # Uncomment for detailed traceback during debugging
        # traceback.print_exc()

    print("\nAnalysis complete.")


# --- Simulation Function (for Testing/Example) --- (No changes needed)
def generate_simulated_data(sim_start_date_str: str = None,
                           sim_end_date_str: str = None,
                           num_days_history: int = 455) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates sample total return and weights DataFrames for testing/example.
    [... full docstring remains the same ...]
    """
    print("--- Generating Simulated Data for Testing ---")

    if sim_end_date_str:
        end_sim_date = pd.to_datetime(sim_end_date_str)
    else:
        end_sim_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

    if sim_start_date_str:
        start_sim_date = pd.to_datetime(sim_start_date_str)
    else:
        start_sim_date = end_sim_date - pd.Timedelta(days=num_days_history - 1)

    dates_tr = pd.date_range(start_sim_date, end_sim_date, freq='D')
    tickers = ['SIM_A', 'SIM_B', 'SIM_C', 'SIM_D', 'SIM_E']

    np.random.seed(42)
    n_days = len(dates_tr)
    daily_ret_sim = np.random.normal(0.0005, 0.015, size=(n_days, len(tickers))) + 1
    tr_index_values = 100 * np.cumprod(daily_ret_sim, axis=0)
    total_return_df = pd.DataFrame(tr_index_values, index=dates_tr, columns=tickers).sort_index()

    weights_list = []
    strategies = ['SimGrowth', 'SimValue']
    weight_change_dates = pd.date_range(start_sim_date, end_sim_date, freq='MS')

    for dt in weight_change_dates:
        w_g = np.random.rand(len(tickers)); w_g /= w_g.sum()
        for i, t in enumerate(tickers):
             if not (t == 'SIM_C' and dt.month % 4 == 0):
                weights_list.append({'date': dt, 'ticker': t, 'strategy': 'SimGrowth', 'weight': w_g[i]})
        val_tickers = ['SIM_A', 'SIM_B', 'SIM_F', 'SIM_G']
        w_v = np.random.rand(len(val_tickers)); w_v /= w_v.sum()
        for i, t in enumerate(val_tickers):
             if not (t == 'SIM_B' and dt.month % 3 == 0):
                 weights_list.append({'date': dt, 'ticker': t, 'strategy': 'SimValue', 'weight': w_v[i]})

    all_weight_tickers = set(item['ticker'] for item in weights_list)
    missing_tickers = list(all_weight_tickers - set(total_return_df.columns))
    if missing_tickers:
        print(f"Adding dummy return data for simulated tickers missing from returns: {missing_tickers}")
        for mticker in missing_tickers:
            ret_sim = np.random.normal(0.0003, 0.018, size=(n_days, 1)) + 1
            tr_sim = 100 * np.cumprod(ret_sim, axis=0)
            total_return_df[mticker] = tr_sim.flatten()
        total_return_df = total_return_df.sort_index(axis=1)

    weights_df = pd.DataFrame(weights_list)
    weights_df['date'] = pd.to_datetime(weights_df['date'])

    print(f"Simulated data generated from {start_sim_date.date()} to {end_sim_date.date()}.")
    return total_return_df, weights_df


# --- Main Execution Block --- (No changes needed)

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
    data_loaded = False # Initialize flag
    return_data = None
    weights_data = None

    try:
        # Attempt to load data from specified paths
        return_data = load_return_data(RETURN_DATA_CSV, index_col=0, parse_dates=True)
        weights_data = load_weights_data(WEIGHTS_DATA_CSV, date_column_name=WEIGHTS_DATE_COLUMN)
        print("\nSuccessfully loaded data from CSV files.")
        data_loaded = True # Set flag if loading succeeds

    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"\nWARNING: Failed to load data from CSV: {e}")
        print("Data loading failed. Check paths/formats or enable simulation fallback.")

        # --- !!! SIMULATION FALLBACK (Uncomment section below to enable) !!! ---
        # print("\nAttempting to run with SIMULATED data for demonstration...")
        try:
             # Generate simulated data
             return_data, weights_data = generate_simulated_data(
                 sim_start_date_str=ANALYSIS_START_DATE, # Use configured dates for sim range if possible
                 sim_end_date_str=ANALYSIS_END_DATE
             )
             # Update strategy names for simulation
             STRATEGY_A = 'SimGrowth'
             STRATEGY_B = 'SimValue'
             # Set flag to True since simulation succeeded
             data_loaded = True
             print("Successfully generated SIMULATED data for fallback analysis.")
        except Exception as sim_e:
             print(f"ERROR: Failed to generate SIMULATED data: {sim_e}")
             data_loaded = False # Ensure flag is false if sim fails too
        # --- !!! END SIMULATION FALLBACK SECTION !!! ---

    except Exception as e:
        # Catch other unexpected errors during initial loading attempt
        print(f"\nUNEXPECTED FATAL ERROR during data loading: {e}")
        data_loaded = False # Ensure flag is false


    # --- Run Analysis ---
    if data_loaded and return_data is not None and weights_data is not None:
        data_source = "SIMULATED" if STRATEGY_A == 'SimGrowth' else "loaded"
        print(f"\nProceeding with analysis using {data_source} data.")

        run_attribution_analysis(
            total_return_df=return_data,
            weights_df=weights_data,
            start_date=ANALYSIS_START_DATE,
            end_date=ANALYSIS_END_DATE,
            strategy_A=STRATEGY_A,
            strategy_B=STRATEGY_B,
            weights_date_col=WEIGHTS_DATE_COLUMN,
            results_suffix=RESULTS_FILE_SUFFIX,
            max_bars_plot=25
        )
    else:
        print("\nAnalysis skipped due to data loading errors.")
        sys.exit(1) # Exit with error status

    print("\n--- Performance Attribution Script Finished ---")