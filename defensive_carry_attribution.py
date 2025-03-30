import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional

# ==============================================================================
# Core Analysis Function with Ticker Mapping
# ==============================================================================

def analyze_fx_carry(
    excel_file_path: str,
    returns_sheet_name: str,
    weights_sheet_name: str,
    mapping_file_path: str, # New: Path to the mapping file
    mapping_sheet_name: str, # New: Sheet name for mapping
    ticker_col_name: str,    # New: Column name for tickers in mapping sheet
    currency_col_name: str,  # New: Column name for currency pairs in mapping sheet
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Analyzes FX carry strategy returns from an Excel file using a ticker mapping.

    It aggregates returns based on two criteria:
    1. Currency Type: G10 vs. Emerging Market (EM), determined by mapping tickers
       (column names in returns/weights sheets) to currency pairs using a separate table.
    2. Position Direction: Short USD (positive weight) vs. Long USD (negative weight).

    Args:
        excel_file_path (str): Path to the Excel file containing returns and weights data.
        returns_sheet_name (str): Name of the sheet with return attribution data.
                                  Assumes Date index, columns are strategy tickers.
        weights_sheet_name (str): Name of the sheet with weight data.
                                  Assumes Date index, columns are strategy tickers matching returns sheet.
        mapping_file_path (str): Path to the Excel file containing the ticker-to-currency mapping.
                                 (Can be the same as excel_file_path).
        mapping_sheet_name (str): Name of the sheet containing the mapping table.
        ticker_col_name (str): The exact name of the column in the mapping sheet
                               that contains the strategy tickers.
        currency_col_name (str): The exact name of the column in the mapping sheet
                                 that contains the corresponding currency pairs (e.g., 'EURUSD').
        start_date (Optional[str]): Optional start date for analysis ('YYYY-MM-DD').
        end_date (Optional[str]): Optional end date for analysis ('YYYY-MM-DD').

    Returns:
        Dict[str, float]: A dictionary containing the aggregated returns (in USD)
                          for each of the four categories. Returns empty if errors occur.
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
        # Validate required columns exist in mapping sheet
        if ticker_col_name not in mapping_df.columns:
            print(f"Error: Ticker column '{ticker_col_name}' not found in mapping sheet '{mapping_sheet_name}'.")
            return {}
        if currency_col_name not in mapping_df.columns:
            print(f"Error: Currency pair column '{currency_col_name}' not found in mapping sheet '{mapping_sheet_name}'.")
            return {}

        # Create the dictionary: Ticker -> Currency Pair
        # Drop rows where ticker or currency pair is missing
        mapping_df = mapping_df.dropna(subset=[ticker_col_name, currency_col_name])
        # Handle potential duplicate tickers - keep the first occurrence
        mapping_df = mapping_df.drop_duplicates(subset=[ticker_col_name], keep='first')
        ticker_to_currency_map: Dict[str, str] = pd.Series(
            mapping_df[currency_col_name].values,
            index=mapping_df[ticker_col_name]
        ).to_dict()
        print(f"Successfully loaded mapping for {len(ticker_to_currency_map)} unique tickers.")
        if not ticker_to_currency_map:
            print("Warning: Ticker mapping dictionary is empty.")
            # Continue, but likely no results will be categorized

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

    # --- Step 2: Data Validation and Alignment ---
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        print(f"Error: Index of sheet '{returns_sheet_name}' is not recognized as dates.")
        return {}
    if not isinstance(weights_df.index, pd.DatetimeIndex):
        print(f"Error: Index of sheet '{weights_sheet_name}' is not recognized as dates.")
        return {}

    # Align based on common tickers (columns) and dates (index)
    common_tickers = returns_df.columns.intersection(weights_df.columns)
    common_index = returns_df.index.intersection(weights_df.index)

    if not common_tickers.tolist():
        print("Error: No common tickers (columns) found between the returns and weights sheets.")
        return {}
    if not common_index.tolist():
         print("Error: No common dates found between the returns and weights sheets.")
         return {}

    returns_df = returns_df.loc[common_index, common_tickers]
    weights_df = weights_df.loc[common_index, common_tickers]
    print(f"Aligned data: {len(common_index)} dates and {len(common_tickers)} common tickers.")

    # --- Step 3: Filtering by Date ---
    print(f"Filtering data for period: {start_date or 'Start'} to {end_date or 'End'}...")
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    if start_dt:
        returns_df = returns_df.loc[returns_df.index >= start_dt]
        weights_df = weights_df.loc[weights_df.index >= start_dt]
    if end_dt:
        returns_df = returns_df.loc[returns_df.index <= end_dt]
        weights_df = weights_df.loc[weights_df.index <= end_dt]

    if returns_df.empty or weights_df.empty:
        print(f"Warning: No data available for the specified date range ({start_date} to {end_date}).")
        return {}
    print(f"Data filtered: {len(returns_df.index)} dates remaining.")


    # --- Step 4: Categorization and Aggregation (Using Ticker Mapping) ---
    print("Categorizing positions using ticker mapping and aggregating returns...")

    # Create boolean masks based on weights (columns are tickers)
    is_short_usd_mask: pd.DataFrame = weights_df > 0
    is_long_usd_mask: pd.DataFrame = weights_df < 0

    # Initialize result variables
    g10_short_usd_total: float = 0.0
    g10_long_usd_total: float = 0.0
    em_short_usd_total: float = 0.0
    em_long_usd_total: float = 0.0
    processed_ticker_count = 0

    # Iterate through each common ticker found in the data
    for ticker in common_tickers:
        # Find the corresponding currency pair from the mapping
        currency_pair = ticker_to_currency_map.get(ticker)

        if currency_pair is None:
            # print(f"Warning: Ticker '{ticker}' not found in mapping table. Skipping.") # Optional: Reduce verbosity
            continue # Skip this ticker if it's not in the map

        processed_ticker_count += 1

        # Determine if the mapped currency pair is G10 or EM
        is_g10 = currency_pair in g10_pairs

        # Get the returns series for this specific ticker
        ticker_returns = returns_df[ticker]

        # Get the boolean masks for position direction for this specific ticker
        ticker_short_mask = is_short_usd_mask[ticker]
        ticker_long_mask = is_long_usd_mask[ticker]

        # Calculate the sum of returns for short USD positions for this ticker
        # .where() keeps returns where mask is True, replaces others with NaN
        # .sum() ignores NaNs and sums the relevant returns
        short_usd_returns = ticker_returns.where(ticker_short_mask).sum()

        # Calculate the sum of returns for long USD positions for this ticker
        long_usd_returns = ticker_returns.where(ticker_long_mask).sum()

        # Add the aggregated returns for this ticker to the correct category total
        if is_g10:
            g10_short_usd_total += short_usd_returns
            g10_long_usd_total += long_usd_returns
        else: # Is EM
            em_short_usd_total += short_usd_returns
            em_long_usd_total += long_usd_returns

    print(f"Processed {processed_ticker_count} tickers found in both data and mapping.")
    if processed_ticker_count == 0 and len(common_tickers) > 0:
        print("Warning: None of the common tickers were found in the mapping table. Results will be zero.")


    # --- Step 5: Format Results ---
    results: Dict[str, float] = {
        "G10 Short USD": g10_short_usd_total,
        "G10 Long USD": g10_long_usd_total,
        "EM Short USD": em_short_usd_total,
        "EM Long USD": em_long_usd_total,
    }

    # Replace potential NaN values with 0.0 (e.g., if a category had no returns)
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
    - Calls the analysis function.
    - Prints the results.
    - Generates and displays a bar chart visualization.
    """

    # --- !!! USER CONFIGURATION REQUIRED !!! ---

    # --- 1. Data File Details ---
    YOUR_EXCEL_FILE = 'path/to/your/fx_data.xlsx' # <--- CHANGE THIS (File with returns/weights)
    RETURNS_SHEET = 'Returns'                     # <--- CHANGE THIS (Sheet name for returns)
    WEIGHTS_SHEET = 'Weights'                     # <--- CHANGE THIS (Sheet name for weights)

    # --- 2. Ticker Mapping Details ---
    # Can be the same file as above or a different one
    MAPPING_FILE = 'path/to/your/mapping_data.xlsx' # <--- CHANGE THIS (File with ticker mapping)
    MAPPING_SHEET = 'Mapping'                       # <--- CHANGE THIS (Sheet name for mapping)
    TICKER_COLUMN = 'Substrategy Ticker'            # <--- CHANGE THIS (Column name for tickers in mapping sheet)
    CURRENCY_COLUMN = 'Currency Pair'               # <--- CHANGE THIS (Column name for currency pairs in mapping sheet)

    # --- 3. Analysis Period (Optional) ---
    # Set to None to analyze the entire dataset found in the file. Use 'YYYY-MM-DD'.
    START_DATE = '2023-01-01' # Example start date (or None)
    END_DATE = '2023-12-31'   # Example end date (or None)
    # START_DATE = None
    # END_DATE = None

    # --- End of User Configuration ---


    # --- Run the analysis using configured parameters ---
    print("="*50)
    print("Starting FX Carry Return Attribution Analysis (with Ticker Mapping)")
    print("="*50)
    aggregated_returns = analyze_fx_carry(
        excel_file_path=YOUR_EXCEL_FILE,
        returns_sheet_name=RETURNS_SHEET,
        weights_sheet_name=WEIGHTS_SHEET,
        mapping_file_path=MAPPING_FILE,      # Pass mapping info
        mapping_sheet_name=MAPPING_SHEET,    # Pass mapping info
        ticker_col_name=TICKER_COLUMN,       # Pass mapping info
        currency_col_name=CURRENCY_COLUMN,   # Pass mapping info
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
        plt.title(f"FX Carry Return Attribution ({period_str})")
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
