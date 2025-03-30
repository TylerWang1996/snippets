import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional

# ==============================================================================
# Core Analysis Function with Ticker Mapping, FFilled & Shifted Weights
# ==============================================================================

def analyze_fx_carry(
    excel_file_path: str,
    returns_sheet_name: str,
    weights_sheet_name: str,
    mapping_file_path: str,
    mapping_sheet_name: str,
    ticker_col_name: str,
    currency_col_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Analyzes FX carry strategy returns using ticker mapping, forward-filled, and lagged weights.

    Handles sparse weight data (only on rebalance dates) by forward-filling.
    Aggregates returns based on:
    1. Currency Type: G10 vs. EM (via ticker mapping).
    2. Position Direction: Short USD (positive weight) vs. Long USD (negative weight),
       using forward-filled weights from the *previous* period (t-1) for returns at period t.

    Args:
        excel_file_path (str): Path to the Excel file with returns/weights data.
        returns_sheet_name (str): Sheet name for return attribution data (Date index, ticker columns).
        weights_sheet_name (str): Sheet name for weight data (Date index, ticker columns, potentially sparse).
        mapping_file_path (str): Path to the Excel file with ticker-to-currency mapping.
        mapping_sheet_name (str): Sheet name for the mapping table.
        ticker_col_name (str): Column name for tickers in the mapping sheet.
        currency_col_name (str): Column name for currency pairs (e.g., 'EURUSD') in the mapping sheet.
        start_date (Optional[str]): Optional start date for analysis ('YYYY-MM-DD').
        end_date (Optional[str]): Optional end date for analysis ('YYYY-MM-DD').

    Returns:
        Dict[str, float]: Dictionary of aggregated returns for the four categories. Empty if errors occur.
    """

    # --- Configuration: Define G10 Currencies ---
    g10_currencies: List[str] = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']
    g10_pairs: List[str] = [f'{curr}USD' for curr in g10_currencies]
    print(f"Defined G10 pairs: {g10_pairs}")

    # --- Step 1a: Load Ticker-to-Currency Mapping ---
    print(f"Loading ticker mapping from: {mapping_file_path} (Sheet: '{mapping_sheet_name}')")
    try:
        mapping_df = pd.read_excel(
            mapping_file_path,
            sheet_name=mapping_sheet_name
        )
        # Basic validation for mapping columns
        if ticker_col_name not in mapping_df.columns or currency_col_name not in mapping_df.columns:
             print(f"Error: Required mapping columns ('{ticker_col_name}', '{currency_col_name}') not found in sheet '{mapping_sheet_name}'.")
             return {}

        mapping_df = mapping_df.dropna(subset=[ticker_col_name, currency_col_name])
        mapping_df = mapping_df.drop_duplicates(subset=[ticker_col_name], keep='first')
        ticker_to_currency_map: Dict[str, str] = pd.Series(
            mapping_df[currency_col_name].values,
            index=mapping_df[ticker_col_name]
        ).to_dict()
        print(f"Successfully loaded mapping for {len(ticker_to_currency_map)} unique tickers.")
        if not ticker_to_currency_map: print("Warning: Ticker mapping dictionary is empty.")

    except FileNotFoundError:
        print(f"Error: Mapping file not found at '{mapping_file_path}'")
        return {}
    except Exception as e:
        print(f"Error reading mapping file '{mapping_file_path}': {e}")
        return {}

    # --- Step 1b: Load Returns and Weights Data ---
    print(f"Loading returns/weights data from: {excel_file_path}")
    try:
        returns_df = pd.read_excel(
            excel_file_path,
            sheet_name=returns_sheet_name,
            index_col=0, parse_dates=True
        )
        weights_df = pd.read_excel(
            excel_file_path,
            sheet_name=weights_sheet_name,
            index_col=0, parse_dates=True
        )
        print(f"Successfully loaded sheets: '{returns_sheet_name}' and '{weights_sheet_name}'")

    except FileNotFoundError:
        print(f"Error: Data file not found at '{excel_file_path}'")
        return {}
    except Exception as e:
        print(f"Error reading data file '{excel_file_path}': {e}")
        return {}

    # --- Step 2: Data Validation, Alignment, Forward Fill, and Shift ---
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        print(f"Error: Index of sheet '{returns_sheet_name}' is not recognized as dates.")
        return {}
    if not isinstance(weights_df.index, pd.DatetimeIndex):
        print(f"Error: Index of sheet '{weights_sheet_name}' is not recognized as dates.")
        return {}

    # Find common tickers BEFORE reindexing weights
    common_tickers = returns_df.columns.intersection(weights_df.columns)
    if not common_tickers.tolist():
        print("Error: No common tickers (columns) found between the returns and weights sheets.")
        return {}
    print(f"Found {len(common_tickers)} common tickers.")

    # Subset to common tickers first
    returns_df = returns_df[common_tickers]
    weights_df = weights_df[common_tickers]

    # --- Handle Sparse Weights: Reindex, Forward Fill, then Shift ---
    print("Aligning weights to return dates, forward-filling, and shifting...")
    # 1. Reindex weights to match the dates in the returns data. Introduces NaNs for missing dates.
    weights_aligned = weights_df.reindex(returns_df.index)
    # 2. Forward fill the NaNs using the last known weight.
    weights_filled = weights_aligned.ffill()
    # 3. Shift the forward-filled weights by 1 period for t-1 logic.
    weights_final = weights_filled.shift(1)
    # The first row(s) will have NaNs either from the original data start or the shift.

    # Now, align both dataframes to ensure they cover the exact same dates and tickers AFTER processing weights
    common_index = returns_df.index.intersection(weights_final.index)
    if not common_index.tolist():
         print("Error: No common dates found after aligning returns and processed weights.")
         return {}

    returns_df = returns_df.loc[common_index]
    weights_final = weights_final.loc[common_index] # Use the fully processed weights
    print(f"Aligned data post-processing: {len(common_index)} dates and {len(common_tickers)} common tickers.")


    # --- Step 3: Filtering by Date ---
    # Filtering is applied AFTER alignment and weight processing
    print(f"Filtering data for period: {start_date or 'Start'} to {end_date or 'End'}...")
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    # Apply date filters if provided
    if start_dt:
        returns_df = returns_df.loc[returns_df.index >= start_dt]
        weights_final = weights_final.loc[weights_final.index >= start_dt]
    if end_dt:
        returns_df = returns_df.loc[returns_df.index <= end_dt]
        weights_final = weights_final.loc[weights_final.index <= end_dt]

    # Check if data remains after filtering
    if returns_df.empty or weights_final.empty:
        print(f"Warning: No data available for the specified date range ({start_date} to {end_date}) after filtering.")
        return {}
    print(f"Data filtered: {len(returns_df.index)} dates remaining.")


    # --- Step 4: Categorization and Aggregation (Using Ticker Mapping & Processed Weights) ---
    print("Categorizing positions using ticker mapping and processed weights, then aggregating returns...")

    # Create boolean masks based on the FINAL processed (reindexed, ffilled, shifted) weights
    is_short_usd_mask: pd.DataFrame = weights_final > 0
    is_long_usd_mask: pd.DataFrame = weights_final < 0

    # Initialize result variables
    g10_short_usd_total: float = 0.0
    g10_long_usd_total: float = 0.0
    em_short_usd_total: float = 0.0
    em_long_usd_total: float = 0.0
    processed_ticker_count = 0

    # Iterate through each common ticker available in the filtered data
    for ticker in returns_df.columns: # Iterate columns of the final filtered dataframe
        currency_pair = ticker_to_currency_map.get(ticker)
        if currency_pair is None:
            continue # Skip if ticker somehow lost mapping (shouldn't happen if filtered correctly)

        processed_ticker_count += 1
        is_g10 = currency_pair in g10_pairs

        # Select returns and masks for the current ticker from the *filtered* DataFrames
        ticker_returns = returns_df[ticker]
        ticker_short_mask = is_short_usd_mask[ticker]
        ticker_long_mask = is_long_usd_mask[ticker]

        # Aggregate returns based on the final processed weight masks
        # NaNs in weights_final (e.g., at the start) result in False masks, correctly excluding these returns.
        short_usd_returns = ticker_returns.where(ticker_short_mask).sum()
        long_usd_returns = ticker_returns.where(ticker_long_mask).sum()

        # Add to the appropriate category total
        if is_g10:
            g10_short_usd_total += short_usd_returns
            g10_long_usd_total += long_usd_returns
        else: # Is EM
            em_short_usd_total += short_usd_returns
            em_long_usd_total += long_usd_returns

    print(f"Processed {processed_ticker_count} tickers found in filtered data and mapping.")
    if processed_ticker_count == 0 and len(returns_df.columns) > 0:
        print("Warning: None of the tickers in the final filtered data were found in the mapping table (or processed). Results may be zero.")


    # --- Step 5: Format Results ---
    results: Dict[str, float] = {
        "G10 Short USD": g10_short_usd_total,
        "G10 Long USD": g10_long_usd_total,
        "EM Short USD": em_short_usd_total,
        "EM Long USD": em_long_usd_total,
    }
    results = {k: v if pd.notna(v) else 0.0 for k, v in results.items()}
    print("Analysis complete.")
    return results

# ==============================================================================
# Script Execution Block
# ==============================================================================

if __name__ == "__main__":
    """
    Main execution block:
    - Configure file paths, sheet names, mapping details, and analysis period.
    - Calls the analysis function (now handles sparse weights via ffill).
    - Prints the results.
    - Generates and displays a bar chart visualization.
    """

    # --- !!! USER CONFIGURATION REQUIRED !!! ---

    # --- 1. Data File Details ---
    YOUR_EXCEL_FILE = 'path/to/your/fx_data.xlsx' # <--- CHANGE THIS
    RETURNS_SHEET = 'Returns'                     # <--- CHANGE THIS
    WEIGHTS_SHEET = 'Weights'                     # <--- CHANGE THIS (Can be sparse)

    # --- 2. Ticker Mapping Details ---
    MAPPING_FILE = 'path/to/your/mapping_data.xlsx' # <--- CHANGE THIS
    MAPPING_SHEET = 'Mapping'                       # <--- CHANGE THIS
    TICKER_COLUMN = 'Substrategy Ticker'            # <--- CHANGE THIS
    CURRENCY_COLUMN = 'Currency Pair'               # <--- CHANGE THIS

    # --- 3. Analysis Period (Optional) ---
    START_DATE = '2023-01-01' # Example start date (or None)
    END_DATE = '2023-12-31'   # Example end date (or None)
    # START_DATE = None
    # END_DATE = None

    # --- End of User Configuration ---


    # --- Run the analysis using configured parameters ---
    print("="*50)
    print("Starting FX Carry Return Attribution Analysis (with Ticker Mapping, FFilled & Shifted Weights)")
    print("="*50)
    aggregated_returns = analyze_fx_carry(
        excel_file_path=YOUR_EXCEL_FILE,
        returns_sheet_name=RETURNS_SHEET,
        weights_sheet_name=WEIGHTS_SHEET,
        mapping_file_path=MAPPING_FILE,
        mapping_sheet_name=MAPPING_SHEET,
        ticker_col_name=TICKER_COLUMN,
        currency_col_name=CURRENCY_COLUMN,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # --- Print & Visualize Results ---
    if aggregated_returns:
        print("\n--- Aggregated FX Carry Returns ---")
        period_str = f"{START_DATE or 'Start'} to {END_DATE or 'End'}"
        print(f"Period: {period_str}")
        print("-" * 40)
        total_return = 0.0
        for category, ret in aggregated_returns.items():
            print(f"{category:<15}: {ret:,.2f}")
            total_return += ret
        print("-" * 40)
        print(f"{'Total':<15}: {total_return:,.2f}")
        print("-" * 40)

        # --- Visualization ---
        print("\nGenerating plot...")
        categories = list(aggregated_returns.keys())
        values = list(aggregated_returns.values())

        plt.figure(figsize=(10, 6))
        colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive']
        bars = plt.bar(categories, values, color=colors)

        plt.ylabel("Aggregated Return (USD)")
        plt.title(f"FX Carry Return Attribution ({period_str}) - Using FFilled Lagged Weights") # Updated title
        plt.xticks(rotation=0)
        plt.axhline(0, color='grey', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            va_pos = 'bottom' if yval >= 0 else 'top'
            plt.text(
                x=bar.get_x() + bar.get_width() / 2.0, y=yval,
                s=f'{yval:,.2f}', va=va_pos, ha='center', fontsize=9
            )

        plt.tight_layout()
        print("Displaying bar chart visualization (close the plot window to exit script)...")
        plt.show()

    else:
        print("\nAnalysis did not produce results. Please check input files, sheet names, column names, dates, and error messages above.")

    print("\nScript finished.")
