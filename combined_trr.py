import pandas as pd
import os # Optional: Used for creating output directory

# --- Configuration ---

# 1. Define the path to your base CSV file
BASE_CSV_PATH = 'path/to/your/base_data.csv' # <--- CHANGE THIS

# 2. Define a list of paths to the other CSV files to combine
OTHER_CSV_PATHS = [
    'path/to/your/other_data_1.csv', # <--- CHANGE THESE
    'path/to/your/other_data_2.csv',
    # Add more paths as needed
]

# 3. Define the start date (inclusive) in 'YYYY-MM-DD' format
START_DATE = '2020-01-01' # <--- CHANGE THIS

# 4. Define the name of the column containing dates in your CSVs
DATE_COLUMN_NAME = 'date' # <--- CHANGE THIS if your date column has a different name

# 5. Define the path for the output file (optional)
OUTPUT_CSV_PATH = 'combined_total_return_index.csv' # <--- CHANGE THIS (or set to None to skip saving)
OUTPUT_DIR = 'output' # Optional: Subdirectory for the output file

# --- Script Logic ---

def combine_total_return_data(base_path, other_paths, start_date_str, date_col):
    """
    Combines total return index data from multiple CSVs based on a base file's dates.

    Args:
        base_path (str): Path to the base CSV file.
        other_paths (list): A list of paths to other CSV files.
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        date_col (str): The name of the date column in the CSVs.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data, or None if an error occurs.
    """
    print(f"Starting data combination process...")
    print(f"Base file: {base_path}")
    print(f"Other files: {other_paths}")
    print(f"Start date: {start_date_str}")
    print(f"Date column: {date_col}")

    try:
        # Convert start date string to Timestamp for comparison
        start_timestamp = pd.to_datetime(start_date_str)

        # --- Load and Filter Base Data ---
        print(f"\nLoading base file: {base_path}")
        base_df = pd.read_csv(base_path, index_col=date_col, parse_dates=True)
        print(f"  Base data loaded with shape: {base_df.shape}")

        # Filter base data by start date and sort index (important!)
        base_df = base_df[base_df.index >= start_timestamp].sort_index()
        print(f"  Base data filtered from {start_date_str}. Shape after filtering: {base_df.shape}")

        if base_df.empty:
            print(f"Error: Base DataFrame is empty after filtering by start date {start_date_str}.")
            print(f"       Please check the base file ('{base_path}') and the start date.")
            return None

        # List to hold all dataframes to be combined (starting with the base)
        all_dfs_to_combine = [base_df]

        # --- Load and Reindex Other Data ---
        for other_path in other_paths:
            try:
                print(f"\nProcessing other file: {other_path}")
                other_df = pd.read_csv(other_path, index_col=date_col, parse_dates=True)
                print(f"  Other data loaded with shape: {other_df.shape}")

                # Sort index before reindexing/ffill for reliable results
                other_df = other_df.sort_index()

                # Reindex using the filtered base_df's index and forward fill
                # This aligns the dates and handles missing values as requested
                print(f"  Reindexing and forward-filling '{os.path.basename(other_path)}' to match base dates...")
                reindexed_other_df = other_df.reindex(base_df.index, method='ffill')
                print(f"  Shape after reindexing: {reindexed_other_df.shape}")

                # Add the reindexed dataframe to our list
                all_dfs_to_combine.append(reindexed_other_df)

            except FileNotFoundError:
                print(f"  Warning: File not found - {other_path}. Skipping.")
            except Exception as e:
                print(f"  Warning: Could not process file {other_path}. Error: {e}. Skipping.")

        # --- Combine DataFrames ---
        print("\nCombining all DataFrames...")
        # Concatenate along columns (axis=1). Pandas aligns based on the index.
        combined_df = pd.concat(all_dfs_to_combine, axis=1)
        print(f"Shape after initial concatenation: {combined_df.shape}")

        # --- Handle Duplicate Columns (Optional but Recommended) ---
        # If the same ticker exists in multiple files, concat creates duplicate columns.
        # This line keeps the *first* occurrence of each column name encountered.
        # Since base_df was added first, its columns take precedence if duplicated later.
        if combined_df.columns.duplicated().any():
            print("Duplicate column names found. Keeping first occurrence...")
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            print(f"Shape after removing duplicate columns: {combined_df.shape}")
        else:
            print("No duplicate column names found.")

        print("\nCombination complete.")
        return combined_df

    except FileNotFoundError:
        print(f"Error: Base file not found - {base_path}. Please check the path.")
        return None
    except KeyError:
         print(f"Error: Date column '{date_col}' not found in one of the files.")
         print(f"       Please ensure all CSVs have a column named '{date_col}'.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    final_data = combine_total_return_data(
        base_path=BASE_CSV_PATH,
        other_paths=OTHER_CSV_PATHS,
        start_date_str=START_DATE,
        date_col=DATE_COLUMN_NAME
    )

    if final_data is not None:
        print("\n--- Combined Data Preview (First 5 rows) ---")
        print(final_data.head())
        print("\n--- Combined Data Preview (Last 5 rows) ---")
        print(final_data.tail())
        print(f"\nFinal combined DataFrame shape: {final_data.shape}")

        # --- Save to CSV (Optional) ---
        if OUTPUT_CSV_PATH:
             try:
                 # Create output directory if it doesn't exist
                 if OUTPUT_DIR:
                     os.makedirs(OUTPUT_DIR, exist_ok=True)
                     output_full_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_PATH)
                 else:
                     output_full_path = OUTPUT_CSV_PATH

                 print(f"\nSaving combined data to: {output_full_path}")
                 final_data.to_csv(output_full_path)
                 print("Data saved successfully.")
             except Exception as e:
                 print(f"Error saving data to CSV: {e}")
    else:
        print("\nFailed to generate combined data.")