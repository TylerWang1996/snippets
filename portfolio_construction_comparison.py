import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch # Added for HRP
from scipy.spatial.distance import squareform # Added for HRP
import warnings
from typing import Dict, List, Tuple, Optional # Keep necessary type hints

# --- Constants for Calculations and Thresholds ---
# Used for annualizing monthly standard deviation
ANNUALIZATION_FACTOR_VOL = np.sqrt(12)
# Used for annualizing monthly covariance
ANNUALIZATION_FACTOR_COV = 12
# Threshold below which volatility is considered zero (for EV weighting)
SMALL_VOL_THRESHOLD = 1e-10
# Threshold for floating point comparisons (e.g., checking near-zero variance/drawdown)
FLOAT_COMPARISON_THRESHOLD = 1e-15
# Tolerance for optimization convergence in ERC calculation
OPTIMIZER_TOLERANCE = 1e-9
# Threshold for verifying ERC result (StDev of % Risk Contributions)
ERC_VERIFICATION_THRESHOLD = 1e-4 # Example: Check if StDev is less than 0.01%
# Small value added to the diagonal of non-Positive Semi-Definite covariance matrices
COV_MATRIX_JITTER = 1e-8

# --- Warning Suppression ---
# Suppress common warnings during optimization or sqrt calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress FutureWarnings often arising from library interactions
warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioConstructor:
    """
    Constructs portfolios using Equal Weight, Equal Volatility, Equal Risk
    Contribution (ERC), and Hierarchical Risk Parity (HRP) methods based on
    monthly Total Return Index (TRI) data.

    Allows specifying different rebalancing frequencies for weight recalculation.

    Attributes:
        returns_df: DataFrame of monthly returns for the underlying assets, derived
                    from the input TRI data. Index is DatetimeIndex.
        tickers: List of asset ticker strings.
        num_assets: Integer count of the assets.
        weights: Dictionary storing the calculated portfolio weights over time for
                 each constructed method. Keys are method names (str), values are
                 DataFrames (index=Date, columns=Tickers). Weights represent the
                 holdings at the *start* of the month.
        returns: Dictionary storing the calculated monthly returns for each
                 constructed portfolio method. Keys are method names (str),
                 values are Series (index=Date).
    """
    def __init__(self, tri_df: pd.DataFrame) -> None:
        """
        Initializes the constructor, validates input, and calculates asset returns.

        Args:
            tri_df: DataFrame with a DatetimeIndex (monthly) and columns for each
                    asset's Total Return Index.

        Raises:
            ValueError: If input DataFrame index is not DatetimeIndex, or if
                        data processing results in an empty DataFrame.
        """
        if not isinstance(tri_df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        # Work on a copy, clean NaNs, and sort by date
        _tri_df = tri_df.copy()
        _tri_df.dropna(axis=0, how='all', inplace=True)
        _tri_df.dropna(axis=1, how='all', inplace=True)
        if _tri_df.empty:
            raise ValueError("Input DataFrame is empty after removing all-NaN rows/columns.")
        _tri_df.sort_index(inplace=True)

        # Calculate monthly percentage returns from TRI
        # Fill potential leading NaNs from pct_change with 0 if needed, though dropna handles it
        self.returns_df = _tri_df.pct_change() #.fillna(0)
        # Drop rows where *any* asset has NaN return (ensures consistent data for cov/corr)
        self.returns_df.dropna(axis=0, how='any', inplace=True)

        if self.returns_df.empty:
             raise ValueError("Return calculation resulted in an empty DataFrame. Ensure TRI data has >1 consecutive valid periods per asset.")

        # Initialize attributes
        self.tickers = self.returns_df.columns.tolist()
        self.num_assets = len(self.tickers)
        self.weights: Dict[str, pd.DataFrame] = {}
        self.returns: Dict[str, pd.Series] = {}

        # User feedback on initialization
        print(f"Initialized PortfolioConstructor with {self.num_assets} assets: {', '.join(self.tickers)}")
        if not self.returns_df.empty:
            print(f"Asset return data available from {self.returns_df.index.min().strftime('%Y-%m-%d')} to {self.returns_df.index.max().strftime('%Y-%m-%d')}")

    def _calculate_portfolio_returns(self, method_name: str, weights_df: pd.DataFrame) -> None:
        """
        Internal method to calculate monthly portfolio returns from asset returns
        and calculated weights, ensuring lookahead bias is avoided.

        Crucially, uses forward-fill (`ffill`) to apply the last known weights
        between rebalance dates specified in weights_df.

        Args:
            method_name: Name of the portfolio construction method (e.g., 'EW').
            weights_df: DataFrame of portfolio weights, indexed ONLY by rebalance dates.
                        These are the target weights decided *at* the index date.
        """
        if weights_df.empty:
             print(f"Warning: Received empty weights DataFrame for '{method_name}'. Cannot calculate returns.")
             self.weights[method_name] = pd.DataFrame(columns=self.tickers)
             self.returns[method_name] = pd.Series(dtype=float, name=method_name)
             return

        # Align weights to full return series index, forward filling between rebalances
        # This is where holding between infrequent rebalances is handled.
        try:
             aligned_weights = weights_df.reindex(self.returns_df.index).ffill()
             # Handle potential NaNs at the beginning before the first weight date
             first_valid_weight_idx = aligned_weights.first_valid_index()
             if first_valid_weight_idx is not None:
                 aligned_weights = aligned_weights.loc[first_valid_weight_idx:]
                 # Reindex again to ensure it matches returns_df after slicing start
                 aligned_weights = aligned_weights.reindex(self.returns_df.index).ffill()
             else: # Should not happen if weights_df is not empty, but safeguard
                 print(f"Warning: Could not find valid weights after aligning for {method_name}.")
                 aligned_weights = pd.DataFrame(0.0, index=self.returns_df.index, columns=self.tickers) # Assign zero weights

        except Exception as e:
             print(f"Error aligning weights for {method_name}: {e}")
             print("Weights Index:", weights_df.index)
             print("Returns Index:", self.returns_df.index)
             self.weights[method_name] = pd.DataFrame(columns=self.tickers)
             self.returns[method_name] = pd.Series(dtype=float, name=method_name)
             return


        # Normalize weights (accounts for potential float issues / ensures sum=1)
        row_sums = aligned_weights.sum(axis=1)
        # Avoid division by zero if a row sum is zero (e.g., after ffill before first weight date)
        safe_row_sums = row_sums.replace(0, 1.0) + FLOAT_COMPARISON_THRESHOLD
        aligned_weights = aligned_weights.div(safe_row_sums, axis=0)
        # Set weights to zero where original sum was zero
        aligned_weights[row_sums.abs() < FLOAT_COMPARISON_THRESHOLD] = 0.0


        # --- Crucial step to avoid lookahead bias ---
        # Shift weights by 1 period: weights decided using data up to time 't'
        # (index date) are applied to returns from 't' to 't+1'.
        shifted_weights = aligned_weights.shift(1)

        # Calculate portfolio returns: Sum of (weight * asset return)
        # Ensure columns match between shifted_weights and returns_df
        common_tickers = self.returns_df.columns.intersection(shifted_weights.columns)
        # Align returns_df to shifted_weights index to handle potential start date mismatches
        aligned_returns, aligned_shifted_weights = self.returns_df.align(shifted_weights, join='inner', axis=0)

        if aligned_returns.empty or aligned_shifted_weights.empty:
            print(f"Warning: No overlapping dates found between returns and shifted weights for {method_name}. Cannot calculate returns.")
            self.weights[method_name] = pd.DataFrame(columns=self.tickers)
            self.returns[method_name] = pd.Series(dtype=float, name=method_name)
            return

        if len(common_tickers) != self.num_assets:
             print(f"Warning: Mismatch in tickers for {method_name} return calculation.")
             portfolio_returns = (aligned_shifted_weights[common_tickers] * aligned_returns[common_tickers]).sum(axis=1)
        else:
             portfolio_returns = (aligned_shifted_weights * aligned_returns).sum(axis=1)


        # Filter results to start only when weights were first available
        # Use the index from the calculated portfolio_returns which is already aligned
        first_calc_date = portfolio_returns.index.min()

        # Store the actual weights *used* for the calculation period (i.e., the shifted weights)
        # Use the aligned index
        effective_weights = aligned_shifted_weights.loc[first_calc_date:].dropna(axis=0, how='all')
        self.weights[method_name] = effective_weights

        # Store the calculated portfolio returns (dropna removes initial NaN from shift if any remain)
        self.returns[method_name] = portfolio_returns.dropna().rename(method_name)

        # User feedback on results
        if not self.returns[method_name].empty:
            print(f"Calculated returns for {method_name} from {self.returns[method_name].index.min().strftime('%Y-%m-%d')} to {self.returns[method_name].index.max().strftime('%Y-%m-%d')}")
        else:
            print(f"No returns were calculated for {method_name} (check data length vs lookback and alignment).")

    def construct_equal_weight(self) -> None:
        """Constructs an Equal Weight (EW) portfolio (1/N weight per asset).
           EW weights are constant and don't require recalculation based on lookback.
        """
        method_name = "EW"
        print(f"\nConstructing {method_name} portfolio...")
        if self.num_assets == 0:
             print(f"Warning: Cannot construct {method_name} with zero assets."); return

        # Calculate equal weight value
        weight_value = 1.0 / self.num_assets
        weights = pd.Series(weight_value, index=self.tickers)

        # Create DataFrame of weights. Since EW is constant, we only need the first date.
        # _calculate_portfolio_returns will forward fill it.
        first_date = self.returns_df.index.min()
        weights_df = pd.DataFrame([weights.values], index=[first_date], columns=self.tickers)

        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")

    def construct_equal_volatility(self,
                                   lookback_years: int = 2,
                                   rebalance_freq_months: int = 1) -> None:
        """
        Constructs an Equal Volatility (EV) portfolio. Weights are inversely
        proportional to each asset's rolling annualized volatility.

        Args:
            lookback_years: Lookback period in years for volatility calculation.
            rebalance_freq_months: Frequency (in months) for recalculating weights (e.g., 1 for monthly, 12 for annual).
        """
        base_method_name = "EV"
        # Add frequency to name if not monthly
        freq_suffix = f"_{rebalance_freq_months}M" if rebalance_freq_months > 1 else ""
        method_name = f"{base_method_name}{freq_suffix}"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years, Rebalance: {rebalance_freq_months} months)...")

        # Validate lookback period
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        if rebalance_freq_months <= 0: raise ValueError("Rebalance frequency must be positive.")
        lookback_window = int(lookback_years * 12)
        min_lookback_required = 2 # Need >= 2 points for std dev
        if lookback_window < min_lookback_required: raise ValueError(f"Lookback window must be >= {min_lookback_required} months.")
        if lookback_window > len(self.returns_df): raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")

        weights_dict = {}
        # Determine first date calculation is possible
        first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index):
            print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return

        first_calc_date = self.returns_df.index[first_calc_date_index]
        first_calc_loc = self.returns_df.index.get_loc(first_calc_date) # Get location of first calc date

        # Iterate through dates where lookback is available
        for date in self.returns_df.loc[first_calc_date:].index:
            current_loc = self.returns_df.index.get_loc(date)
            # Check if it's a rebalance month based on frequency
            months_since_start = current_loc - first_calc_loc
            if months_since_start % rebalance_freq_months == 0:
                # Get data slice for lookback ending at current date 't'
                start_loc = current_loc - lookback_window + 1
                returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]

                # Check if slice has enough data points after potential NaNs
                if returns_slice.dropna().shape[0] < min_lookback_required:
                     print(f"Warning [{date.date()}]: Insufficient non-NaN data points in lookback window for EV. Skipping rebalance.")
                     continue # Skip this rebalance date if not enough data

                # Calculate annualized volatility (std dev * sqrt(12))
                rolling_vol = returns_slice.std(ddof=1) * ANNUALIZATION_FACTOR_VOL # ddof=1 for sample std dev

                # Handle near-zero volatility to avoid division by zero
                rolling_vol = rolling_vol.fillna(SMALL_VOL_THRESHOLD) # Fill NaN std dev (e.g., constant returns)
                rolling_vol[rolling_vol.abs() < SMALL_VOL_THRESHOLD] = SMALL_VOL_THRESHOLD

                # Calculate inverse volatility weights: w_i = (1/vol_i) / sum(1/vol_j)
                inv_vol = 1.0 / rolling_vol
                weights = inv_vol / (inv_vol.sum() + FLOAT_COMPARISON_THRESHOLD) # Add epsilon for safety
                # Re-normalize for safety (handles potential float precision issues)
                weights = weights / (weights.sum() + FLOAT_COMPARISON_THRESHOLD)
                weights_dict[date] = weights # Store weights only for rebalance dates

        if not weights_dict:
             print(f"Warning: No weights were calculated for {method_name}."); return

        # Create DataFrame (will only contain rows for rebalance dates)
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers)
        weights_df.fillna(0, inplace=True) # Should not be needed if calculation skips, but safe

        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")

    # --- ERC Methods ---
    @staticmethod
    def _erc_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Objective function for ERC optimization. Minimizes the variance of
        *percentage* risk contributions: Var( RC_i / TotalRisk ), where
        RC_i = w_i * (Cov * w)_i.
        """
        portfolio_var = weights.T @ cov_matrix @ weights
        # Handle cases of zero portfolio variance (e.g., all zero returns)
        if portfolio_var < FLOAT_COMPARISON_THRESHOLD: return 0.0

        # Marginal Risk Contribution (MRC): How portfolio variance changes with w_i
        mrc = cov_matrix @ weights
        # Percentage Risk Contribution (each asset's share of total portfolio variance)
        percentage_rc = (weights * mrc) / portfolio_var

        # Return the variance of these percentage contributions
        return np.var(percentage_rc)

    def _calculate_erc_weights(self, cov_matrix: pd.DataFrame, calculation_date: pd.Timestamp) -> pd.Series:
        """
        Internal helper to calculate Equal Risk Contribution weights using optimization.
        Also verifies if the resulting weights achieve near-equal risk contributions.

        Args:
            cov_matrix: The annualized covariance matrix (pd.DataFrame).
            calculation_date: The date for which weights are being calculated (for logging).

        Returns:
            pd.Series containing the optimized weights, indexed by ticker.
            Returns Equal Weights if optimization fails. Issues warning if verification fails.
        """
        # --- Input Validation and Alignment ---
        if not isinstance(cov_matrix, pd.DataFrame):
             raise TypeError(f"[{calculation_date.date()}] Covariance matrix must be a DataFrame.")
        if cov_matrix.shape != (self.num_assets, self.num_assets):
             raise ValueError(f"[{calculation_date.date()}] Cov matrix shape mismatch. Expected ({self.num_assets},{self.num_assets}), got {cov_matrix.shape}")
        # Ensure matrix columns/index match self.tickers order
        if not np.all(cov_matrix.columns == self.tickers) or not np.all(cov_matrix.index == self.tickers):
             try:
                 cov_matrix = cov_matrix.reindex(index=self.tickers, columns=self.tickers)
             except Exception as e:
                 raise ValueError(f"[{calculation_date.date()}] Failed to reindex cov matrix: {e}")
             if cov_matrix.isnull().values.any():
                 raise ValueError(f"[{calculation_date.date()}] Cov matrix has NaNs after reindexing.")

        cov_matrix_values = cov_matrix.values # Use numpy array for optimization

        # --- PSD Check & Jitter ---
        try:
            np.linalg.cholesky(cov_matrix_values)
            # Matrix is PSD
        except np.linalg.LinAlgError:
            # Matrix is not PSD, add jitter
            print(f"Warning [{calculation_date.date()}]: Covariance matrix not PSD. Adding diagonal jitter.")
            cov_matrix_values += np.eye(self.num_assets) * COV_MATRIX_JITTER

        # --- Optimization Setup ---
        w0 = np.ones(self.num_assets) / self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        # --- Run Optimizer ---
        result = minimize(
            self._erc_objective,
            w0,
            args=(cov_matrix_values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=OPTIMIZER_TOLERANCE,
            options={'maxiter': 1000, 'ftol': OPTIMIZER_TOLERANCE, 'disp': False} # disp=False to suppress optimizer output
        )

        # --- Handle Results ---
        if result.success:
            # --- Verification Step ---
            optimized_weights_raw = result.x
            portfolio_var = optimized_weights_raw.T @ cov_matrix_values @ optimized_weights_raw

            if portfolio_var > FLOAT_COMPARISON_THRESHOLD:
                mrc = cov_matrix_values @ optimized_weights_raw
                percentage_rc = (optimized_weights_raw * mrc) / portfolio_var
                rc_std_dev = np.std(percentage_rc)

                if rc_std_dev > ERC_VERIFICATION_THRESHOLD:
                    print(
                        f"Warning [{calculation_date.date()}]: ERC optimization reported success, "
                        f"but risk contribution dispersion exceeds threshold."
                    )
                    print(
                        f"  Target StDev(%RC) < {ERC_VERIFICATION_THRESHOLD:.1e}, "
                        f"Actual StDev(%RC): {rc_std_dev:.6e}"
                    )
            elif self.num_assets > 0:
                 pass # Verification passes implicitly if variance is zero
            # --- End Verification Step ---

            # Normalize final weights for precision
            final_weights = result.x / (result.x.sum() + FLOAT_COMPARISON_THRESHOLD)
            return pd.Series(final_weights, index=self.tickers)

        else:
            # Fallback to Equal Weight if optimization fails
            print(f"Warning [{calculation_date.date()}]: ERC optimization failed: {result.message}. Falling back to EW.")
            return pd.Series(w0, index=self.tickers) # Return EW weights

    def construct_erc(self,
                      lookback_years: int = 2,
                      use_ewm: bool = True,
                      rebalance_freq_months: int = 1) -> None:
        """
        Constructs an Equal Risk Contribution (ERC) portfolio using optimization.

        Args:
            lookback_years: Lookback period in years for covariance calculation.
            use_ewm: If True, use Exponentially Weighted Moving Covariance;
                     otherwise, use Simple Moving Covariance.
            rebalance_freq_months: Frequency (in months) for recalculating weights (e.g., 1 for monthly, 12 for annual).
        """
        base_method_name = f"ERC_{'EWM' if use_ewm else 'Simple'}"
        # Add frequency to name if not monthly
        freq_suffix = f"_{rebalance_freq_months}M" if rebalance_freq_months > 1 else ""
        method_name = f"{base_method_name}{freq_suffix}"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years, Rebalance: {rebalance_freq_months} months)...")


        # Validate lookback period
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        if rebalance_freq_months <= 0: raise ValueError("Rebalance frequency must be positive.")
        lookback_window = int(lookback_years * 12)
        # Need at least N+1 points for a non-singular covariance matrix
        min_lookback_required = self.num_assets + 1
        if lookback_window < min_lookback_required:
             raise ValueError(f"Lookback window ({lookback_window}) should be >= {min_lookback_required} months for ERC.")
        if lookback_window > len(self.returns_df):
             raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")


        weights_dict = {}
        # Determine first date calculation is possible
        first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index):
            print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return

        first_calc_date = self.returns_df.index[first_calc_date_index]
        first_calc_loc = self.returns_df.index.get_loc(first_calc_date) # Get location of first calc date

        # Iterate through dates where lookback is available
        for date in self.returns_df.loc[first_calc_date:].index:
            current_loc = self.returns_df.index.get_loc(date)
            # Check if it's a rebalance month based on frequency
            months_since_start = current_loc - first_calc_loc
            if months_since_start % rebalance_freq_months == 0:
                # Get data slice for lookback ending at current date 't'
                start_loc = current_loc - lookback_window + 1
                returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]

                cov_matrix = None
                cov_calc_successful = False
                # Calculate annualized covariance matrix (Simple or EWM)
                try:
                     # Check for sufficient non-NaN data *before* calculating cov
                     if returns_slice.dropna().shape[0] < min_lookback_required:
                         print(f"Warning [{date.date()}]: Insufficient non-NaN data points in lookback window for ERC cov. Skipping rebalance.")
                         continue # Skip this rebalance date

                     if use_ewm:
                         ewm_cov_obj = returns_slice.ewm(
                             span=lookback_window,
                             min_periods=lookback_window,
                             adjust=True
                         )
                         cov_matrix_ewm = ewm_cov_obj.cov(pairwise=True) * ANNUALIZATION_FACTOR_COV
                         if date in cov_matrix_ewm.index.get_level_values(0):
                              cov_matrix = cov_matrix_ewm.loc[date]
                         else:
                              print(f"Warning [{date.date()}]: Could not extract EWM cov matrix for the target date. Skipping rebalance.")
                              continue
                     else:
                         cov_matrix = returns_slice.cov(ddof=1) * ANNUALIZATION_FACTOR_COV

                     # Validate calculated matrix
                     if isinstance(cov_matrix, pd.DataFrame) and not cov_matrix.isnull().values.any() and cov_matrix.shape == (self.num_assets, self.num_assets):
                         cov_calc_successful = True
                     else:
                         if cov_matrix is not None:
                              print(f"Warning [{date.date()}]: Invalid cov matrix ({'EWM' if use_ewm else 'Simple'}). Skipping rebalance.")

                except Exception as e:
                    print(f"Error [{date.date()}] calculating cov matrix: {e}. Skipping rebalance.")

                # Calculate weights only if covariance matrix is valid
                if cov_calc_successful:
                    try:
                        erc_weights = self._calculate_erc_weights(cov_matrix, date)
                        weights_dict[date] = erc_weights # Store weights only for rebalance dates
                    except Exception as e:
                        print(f"Error [{date.date()}] calculating ERC weights: {e}. Skipping rebalance.")


        if not weights_dict:
             print(f"Warning: No weights calculated for {method_name}."); return

        # Create DataFrame (will only contain rows for rebalance dates)
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers)
        weights_df.fillna(0, inplace=True) # Should not be needed

        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")


    # --- HRP Helper Functions (Adapted from Lopez de Prado) ---
    @staticmethod
    def _correlDist(corr: pd.DataFrame) -> np.ndarray:
        """
        Calculates a distance matrix based on correlation.
        Result is the condensed distance matrix required by sch.linkage.
        Args:
            corr: Correlation matrix (pd.DataFrame).
        Returns:
            np.ndarray: Condensed distance matrix.
        """
        # Ensure correlation matrix is square
        if corr.shape[0] != corr.shape[1]:
             raise ValueError("Correlation matrix must be square.")
        # Calculate the distance matrix D = sqrt(0.5 * (1 - C))
        dist_sq = ((1 - corr.fillna(0)) / 2.)**0.5 # Fill NaNs just in case
        # Return the condensed distance matrix (upper triangle as a 1D array)
        return squareform(dist_sq.values, checks=False)

    @staticmethod
    def _getIVP(cov: pd.DataFrame) -> np.ndarray:
        """
        Compute the Inverse-Variance Portfolio weights for assets within a cluster.
        Args:
            cov: Covariance matrix (pd.DataFrame) of assets in the cluster.
        Returns:
            np.ndarray of weights.
        """
        # Extract diagonal and ensure it's a writable copy
        diag_vars = np.diag(cov.values).copy() # Force a writable copy
        # Handle zero variance case using np.abs()
        diag_vars[np.abs(diag_vars) < FLOAT_COMPARISON_THRESHOLD] = FLOAT_COMPARISON_THRESHOLD # Modify the copy
        ivp = 1. / diag_vars
        ivp /= ivp.sum() # Normalize weights within the cluster
        return ivp

    @staticmethod
    def _getClusterVar(cov: pd.DataFrame, cItems: List[str]) -> float:
        """
        Compute variance per cluster using Inverse-Variance weights.
        Args:
            cov: Full covariance matrix (assets ordered by original tickers).
            cItems: List of tickers (strings) in the cluster.
        Returns:
            Cluster variance (float).
        """
        # Slice covariance matrix for the cluster
        cov_ = cov.loc[cItems, cItems]
        # Calculate IVP weights for the cluster
        w_ = PortfolioConstructor._getIVP(cov_).reshape(-1, 1)
        # Calculate cluster variance: w'.Cov.w
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    @staticmethod
    def _getQuasiDiag(link: np.ndarray) -> List[int]:
        """
        Sort clustered items by distance to achieve quasi-diagonalization.
        Args:
            link: Linkage matrix from scipy.cluster.hierarchy.linkage.
        Returns:
            List of original item indices (0 to N-1) in the sorted order.
        """
        link = link.astype(int)
        # Sort from the top cluster down
        sortIx = pd.Series([link[-1, 0], link[-1, 1]]) # Indices of the two items/subclusters in the top cluster
        numItems = link[-1, 3] # Total number of original items

        while sortIx.max() >= numItems:
            # Find clusters (indices >= numItems indicate clusters formed during linkage)
            cluster_indices = sortIx[sortIx >= numItems]
            if cluster_indices.empty: # Should not happen if sortIx.max() >= numItems but check anyway
                 break
            i = cluster_indices.index # Index positions in sortIx Series where clusters are found
            j = cluster_indices.values - numItems # Map cluster index to linkage matrix row index

            # Replace cluster index with its constituent items/sub-clusters from the linkage matrix
            sortIx.loc[i] = link[j, 0] # Item 1 from the linkage row
            # Create Series for Item 2 to append with correct index alignment (needs unique index)
            # Use i + 0.5 temporarily to avoid index collision before sorting
            df0 = pd.Series(link[j, 1], index=i + 0.5)
            sortIx = pd.concat([sortIx, df0]) # Append Item 2
            sortIx = sortIx.sort_index() # Re-sort by index (interleaves item1 and item2)
            sortIx.index = range(sortIx.shape[0]) # Re-index to sequential integers (0, 1, 2, ...)

        return sortIx.astype(int).tolist() # Return as list of integers

    @staticmethod
    def _getRecBipart(cov: pd.DataFrame, sortIx: List[str]) -> pd.Series:
        """
        Recursively compute HRP allocations based on cluster variances.
        Args:
            cov: Covariance matrix (ordered by sortIx).
            sortIx: List of tickers in quasi-diagonal order.
        Returns:
            pd.Series of HRP weights, indexed by tickers in sortIx order.
        """
        w = pd.Series(1.0, index=sortIx) # Start with equal weights
        cItems = [sortIx] # Initialize with all items in one cluster

        while len(cItems) > 0:
            # Bisection: Split clusters into two halves
            next_cItems = []
            for item_list in cItems:
                if len(item_list) > 1:
                    mid = len(item_list) // 2 # Integer division for midpoint index
                    next_cItems.append(item_list[:mid]) # First half
                    next_cItems.append(item_list[mid:]) # Second half
            cItems = next_cItems # Update list of clusters for next level

            if not cItems: break # Exit if no more clusters to split

            # Process adjacent pairs of clusters resulting from the bisection
            for i in range(0, len(cItems), 2):
                # Ensure there's a pair to process
                if i + 1 >= len(cItems): continue

                cItems0 = cItems[i]     # Cluster 1 items (list of tickers)
                cItems1 = cItems[i + 1] # Cluster 2 items (list of tickers)

                # Calculate variance for each cluster using IVP weights within the cluster
                # Note: cov passed here should be the one ordered by sortIx initially
                cVar0 = PortfolioConstructor._getClusterVar(cov, cItems0)
                cVar1 = PortfolioConstructor._getClusterVar(cov, cItems1)

                # Calculate allocation factor (alpha) based on inverse variance ratio
                # This determines how weight is split between the two clusters
                total_cluster_var = cVar0 + cVar1
                if total_cluster_var < FLOAT_COMPARISON_THRESHOLD :
                    # Handle case where both cluster variances are near zero
                    alpha = 0.5
                else:
                    alpha = 1 - (cVar0 / total_cluster_var) # Proportion allocated to cluster 1

                # Update weights recursively: Multiply weights within each cluster by the allocation factor
                w[cItems0] *= alpha       # Weight allocated to cluster 1
                w[cItems1] *= (1 - alpha) # Weight allocated to cluster 2

        # Ensure weights sum to 1 after recursion (handle potential float inaccuracies)
        w /= (w.sum() + FLOAT_COMPARISON_THRESHOLD)
        return w

    # --- End HRP Helper Functions ---


    def construct_hrp(self,
                      lookback_years: int = 2,
                      linkage_method: str = 'ward',
                      rebalance_freq_months: int = 1) -> None:
        """
        Constructs a Hierarchical Risk Parity (HRP) portfolio.

        Args:
            lookback_years: Lookback period in years for covariance/correlation calculation.
            linkage_method: The linkage algorithm to use for hierarchical clustering
                            (e.g., 'ward', 'single', 'average', 'complete'). Default is 'ward'.
            rebalance_freq_months: Frequency (in months) for recalculating weights (e.g., 1 for monthly, 12 for annual).
        """
        base_method_name = f"HRP_{linkage_method}"
        # Add frequency to name if not monthly
        freq_suffix = f"_{rebalance_freq_months}M" if rebalance_freq_months > 1 else ""
        method_name = f"{base_method_name}{freq_suffix}"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years, Rebalance: {rebalance_freq_months} months)...")


        # Validate lookback period
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        if rebalance_freq_months <= 0: raise ValueError("Rebalance frequency must be positive.")
        lookback_window = int(lookback_years * 12)
        # HRP needs enough data for stable cov/corr, N+1 is minimum for non-singular cov
        min_lookback_required = self.num_assets + 1
        if lookback_window < min_lookback_required:
            # Changed to warning instead of error, but calculation might be unstable
            print(f"Warning: Lookback window ({lookback_window}) is less than recommended ({min_lookback_required}) for HRP.")
        if lookback_window > len(self.returns_df):
            raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")

        weights_dict = {}
        # Determine first date calculation is possible
        first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index):
            print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return

        first_calc_date = self.returns_df.index[first_calc_date_index]
        first_calc_loc = self.returns_df.index.get_loc(first_calc_date) # Get location of first calc date

        # Iterate through dates where lookback is available
        for date in self.returns_df.loc[first_calc_date:].index:
            current_loc = self.returns_df.index.get_loc(date)
            # Check if it's a rebalance month based on frequency
            months_since_start = current_loc - first_calc_loc
            if months_since_start % rebalance_freq_months == 0:
                # Get data slice for lookback ending at current date 't'
                start_loc = current_loc - lookback_window + 1
                returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]

                # Calculate NON-ANNUALIZED Covariance and Correlation for HRP internals
                cov_matrix = None
                corr_matrix = None
                matrices_calculated = False
                try:
                    # Check for sufficient non-NaN data points per asset in the slice
                    if returns_slice.dropna().shape[0] < min_lookback_required:
                         print(f"Warning [{date.date()}]: Insufficient non-NaN data points ({returns_slice.dropna().shape[0]} < {min_lookback_required}) in lookback window. Skipping HRP rebalance.")
                         continue # Skip this rebalance date

                    # Calculate matrices using sample statistics (ddof=1)
                    cov_matrix = returns_slice.cov(ddof=1)
                    corr_matrix = returns_slice.corr()

                    # Basic validation
                    if (isinstance(cov_matrix, pd.DataFrame) and not cov_matrix.isnull().values.any() and
                        isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.isnull().values.any() and
                        cov_matrix.shape == (self.num_assets, self.num_assets) and
                        corr_matrix.shape == (self.num_assets, self.num_assets)):
                        matrices_calculated = True
                    else:
                        print(f"Warning [{date.date()}]: Invalid cov or corr matrix calculated (NaNs or shape mismatch). Skipping HRP rebalance.")

                except Exception as e:
                    print(f"Error [{date.date()}] calculating cov/corr matrix: {e}. Skipping HRP rebalance.")

                # Proceed only if matrices are valid
                if matrices_calculated:
                    try:
                        # --- HRP Steps ---
                        dist_condensed = self._correlDist(corr_matrix)
                        link = sch.linkage(dist_condensed, method=linkage_method)
                        sortIx_indices = self._getQuasiDiag(link)
                        sortIx_tickers = cov_matrix.index[sortIx_indices].tolist()
                        cov_matrix_sorted = cov_matrix.loc[sortIx_tickers, sortIx_tickers]
                        hrp_weights_sorted = self._getRecBipart(cov_matrix_sorted, sortIx_tickers)
                        hrp_weights = hrp_weights_sorted.reindex(self.tickers).fillna(0)
                        hrp_weights /= (hrp_weights.sum() + FLOAT_COMPARISON_THRESHOLD)
                        weights_dict[date] = hrp_weights # Store weights only for rebalance dates
                        # --- End HRP Steps ---
                    except Exception as e:
                        print(f"Error [{date.date()}] during HRP calculation steps: {e}. Skipping rebalance.")


        if not weights_dict:
            print(f"Warning: No weights calculated for {method_name}."); return

        # Create DataFrame (will only contain rows for rebalance dates)
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers)
        weights_df.fillna(0, inplace=True) # Should not be needed

        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")


    def get_returns(self) -> pd.DataFrame:
        """Returns DataFrame of monthly returns for all constructed portfolios."""
        if not self.returns: print("Warning: No portfolio returns calculated."); return pd.DataFrame()
        return pd.DataFrame(self.returns)

    def get_weights(self, method_name: str) -> pd.DataFrame:
        """
        Returns DataFrame of portfolio weights for a specific method.

        Args:
            method_name: The name of the method (e.g., 'EW', 'ERC_EWM', 'HRP_ward_12M').

        Returns:
            DataFrame of weights (index=Date, columns=Tickers). These are the
            effective weights used in the return calculation (i.e., shifted and ffilled).

        Raises:
            KeyError: If the method name is not found.
        """
        if method_name not in self.weights:
            available = list(self.weights.keys())
            raise KeyError(f"Weights for '{method_name}' not found. Available: {available}")
        return self.weights[method_name]

    def get_raw_weights(self, method_name: str) -> Optional[pd.DataFrame]:
        """
        EXPERIMENTAL: Attempts to retrieve the raw, unshifted, non-ffilled weights
        as calculated by the construction method *on rebalance dates only*.

        Note: This relies on reconstructing from the `weights_dict` used internally,
              which is not a standard attribute. Use with caution. Returns None if
              the method doesn't use a weights_dict or calculation failed.

        Args:
            method_name: The name of the method (e.g., 'ERC_EWM_12M').

        Returns:
            Optional[pd.DataFrame]: DataFrame of raw weights indexed by rebalance date,
                                    or None if not applicable/available.
        """
        # This is a simplified example; a robust implementation would require
        # storing the raw weights_dict explicitly during construction.
        # For demonstration, we assume the method name implies the parameters.
        print(f"\nAttempting to retrieve raw weights for {method_name} (Experimental)...")
        temp_weights_dict = {}
        params = method_name.split('_')
        base_method = params[0]

        # --- This part needs actual implementation based on stored raw weights ---
        # --- The current code doesn't store the raw dicts persistently ---
        # --- We would need to modify the construct methods to store them ---
        # Example placeholder:
        if method_name in self.weights: # Check if effective weights exist
             print("Warning: Returning effective weights. Raw weights retrieval not fully implemented.")
             # Find the dates where weights actually changed significantly (proxy for rebalance)
             eff_weights = self.weights[method_name]
             # Check where weights differ from the previous row (ignoring first row)
             diff_check = eff_weights.diff().abs().sum(axis=1) > FLOAT_COMPARISON_THRESHOLD
             diff_check.iloc[0] = True # Always include the first date
             raw_weights_approx = eff_weights[diff_check]
             return raw_weights_approx
        else:
             print(f"Raw weights not available for {method_name}.")
             return None
        # --- End Placeholder ---


    def get_total_return_indexes(self, base_value: int = 100) -> pd.DataFrame:
        """
        Calculates Total Return Indexes for all constructed portfolios, starting
        from a common date and base value.

        Args:
            base_value: The starting value for the index (default: 100).

        Returns:
            DataFrame containing the TRI for each portfolio strategy, indexed by date.
            Returns empty DataFrame if calculation is not possible.
        """
        portfolio_returns_df = self.get_returns()
        if portfolio_returns_df.empty: print("Cannot generate TRI: No portfolio returns."); return pd.DataFrame()

        # Exclude portfolios that might have failed entirely (all NaNs)
        valid_returns = portfolio_returns_df.dropna(axis=1, how='all')
        if valid_returns.empty: print("Cannot generate TRI: All portfolios have NaN returns."); return pd.DataFrame()

        # Find first date where ALL remaining portfolios have valid return data
        try:
            # dropna() finds rows where *all* columns are non-NaN
            common_start_date = valid_returns.dropna(axis=0, how='any').index.min()
        except ValueError: # Handles case where dropna() makes the DataFrame empty
            print("Cannot generate TRI: No common start date found across portfolios where all have data."); return pd.DataFrame()

        # Filter returns from the common start date
        common_returns = valid_returns.loc[common_start_date:]
        if common_returns.empty: print(f"Cannot generate TRI: No common returns data from {common_start_date}."); return pd.DataFrame()
        # Fill any remaining NaNs within the common period with 0 (assumes no return for that month)
        common_returns = common_returns.fillna(0)


        # Calculate cumulative product and scale to base value
        # Add 1 before cumprod
        tri = (1 + common_returns).cumprod() * base_value

        # --- Prepend Base Value Row ---
        # Create a row with the base value dated one period before the common start date
        # Use frequency inference if possible, otherwise assume monthly offset
        try:
             freq = pd.infer_freq(common_start_date) or pd.DateOffset(months=1)
             base_date = common_start_date - freq
        except TypeError: # Handle cases where index might not have frequency info
             base_date = common_start_date - pd.Timedelta(days=30) # Approximate month start

        base_row = pd.DataFrame({col: float(base_value) for col in tri.columns}, index=[base_date])

        # Combine base row with the calculated TRI and sort
        final_tri = pd.concat([base_row, tri]).sort_index()

        return final_tri.round(4) # Round for display


class PortfolioAnalyzer:
    """
    Analyzes and compares performance metrics for portfolio strategies and
    underlying assets based on their monthly returns.

    Attributes:
        all_returns: DataFrame containing monthly returns for both portfolios
                     and assets over their common date range.
        common_start_date: The first date used in the analysis period.
        common_end_date: The last date used in the analysis period.
        metrics: List of performance metric names calculated by the analyzer.
    """
    def __init__(self,
                 portfolio_returns: pd.DataFrame,
                 asset_returns: pd.DataFrame):
        """
        Initializes the analyzer, combining returns and finding common date range.

        Args:
            portfolio_returns: DataFrame of monthly portfolio returns.
            asset_returns: DataFrame of monthly asset returns.

        Raises:
            ValueError: If input data is unusable or no common period exists.
        """
        if portfolio_returns.empty and asset_returns.empty:
            raise ValueError("Both portfolio and asset returns are empty.")

        # Combine all returns; handle potential duplicate column names if assets included in portfolios
        self.all_returns = pd.concat([portfolio_returns, asset_returns], axis=1)
        # Drop duplicate columns (e.g., if an asset return was passed in both inputs)
        self.all_returns = self.all_returns.loc[:,~self.all_returns.columns.duplicated()]


        # Drop columns that are entirely NaN
        self.all_returns.dropna(axis=1, how='all', inplace=True)
        if self.all_returns.empty:
            raise ValueError("Combined returns are empty after dropping all-NaN columns.")

        # Determine common analysis period (where all included series have data)
        try:
            # Find first/last date where *at least one* series has data
            first_valid_overall = self.all_returns.dropna(how='all').index.min()
            last_valid_overall = self.all_returns.dropna(how='all').index.max()

            # Slice to this maximal range first
            self.all_returns = self.all_returns.loc[first_valid_overall:last_valid_overall]

            # Now find the common start/end based on rows where *all* columns are non-NaN
            common_data = self.all_returns.dropna(axis=0, how='any')
            if common_data.empty:
                 raise ValueError("No single date where all assets/strategies have non-NaN returns.")

            self.common_start_date = common_data.index.min()
            self.common_end_date = common_data.index.max() # Use max of common period

            # Final slice to the strictly common period where all have data
            self.all_returns = self.all_returns.loc[self.common_start_date:self.common_end_date]

            # Final check if slicing resulted in empty df (e.g., only one common date)
            if self.all_returns.dropna(axis=0, how='any').empty:
                 raise ValueError("No overlapping non-NaN data period found.")

        except Exception as e:
             raise ValueError(f"Error determining common date range or slicing returns: {e}")

        print(f"\nInitializing Analyzer. Common analysis period: {self.common_start_date.strftime('%Y-%m-%d')} to {self.common_end_date.strftime('%Y-%m-%d')}")
        print(f"Analyzing series: {', '.join(self.all_returns.columns)}")


        # Define the metrics to be calculated
        self.metrics = [
            "Annualized Return", "Annualized Volatility", "Information Ratio",
            "Max Drawdown", "Calmar Ratio", "Max Drawdown Recovery (Months)",
            "% Positive Months", "Avg Positive Month Ret", "Avg Negative Month Ret"
        ]

    def _calculate_annualized_return(self, returns_series: pd.Series) -> float:
        """Calculates geometric annualized return from monthly returns."""
        if returns_series.empty or len(returns_series) < 1: return np.nan
        num_months = len(returns_series)
        # Avoid issues with very short periods; require at least 1 month
        if num_months == 0: return np.nan

        # Geometric mean formula: (product(1+r)^(12/num_months)) - 1
        # Handle returns <= -100%, which invalidate geometric linking
        if ((1 + returns_series) <= FLOAT_COMPARISON_THRESHOLD).any():
            # If total loss occurs, geometric mean is undefined or -100%
            # Find first occurrence of near -100% return
            first_loss_idx = np.where((1 + returns_series) <= FLOAT_COMPARISON_THRESHOLD)[0][0]
            if first_loss_idx == 0: return -1.0 # Loss in first period
            # Calculate return up to the point of loss
            effective_returns = returns_series.iloc[:first_loss_idx]
            if effective_returns.empty: return -1.0 # Should not happen if first_loss_idx > 0
            num_effective_months = len(effective_returns)
            # Avoid division by zero if num_effective_months is 0
            if num_effective_months == 0: return -1.0
            return ((1 + effective_returns).prod())**(12.0 / num_effective_months) - 1

        # Standard calculation
        return ((1 + returns_series).prod())**(12.0 / num_months) - 1


    def _calculate_annualized_volatility(self, returns_series: pd.Series) -> float:
        """Calculates annualized volatility (std dev) from monthly returns."""
        # Need at least 2 data points to calculate std dev
        if returns_series.empty or len(returns_series) < 2: return np.nan
        return returns_series.std(ddof=1) * ANNUALIZATION_FACTOR_VOL # Use sample std dev

    def _calculate_information_ratio(self, ann_return: float, ann_vol: float) -> float:
        """
        Calculates Information Ratio (Annualized Return / Annualized Volatility).
        Assumes risk-free rate is zero (can be adapted).
        """
        # Handle zero, near-zero, or NaN volatility
        if ann_vol is None or abs(ann_vol) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(ann_vol):
            return np.nan
        # Avoid division by zero explicitly
        if abs(ann_vol) < FLOAT_COMPARISON_THRESHOLD:
             return np.nan # Or potentially np.inf or 0 depending on context/return sign
        return ann_return / ann_vol

    def _calculate_max_drawdown(self, returns_series: pd.Series) -> Tuple[float, Optional[float]]:
        """
        Calculates the Maximum Drawdown (MDD) and the time (in months) taken
        to recover from that specific drawdown.

        Returns:
            Tuple[float, Optional[float]]: (Max Drawdown value, Recovery duration in months).
                                           Duration is NaN if not recovered by end of series.
        """
        if returns_series.empty or len(returns_series) < 1: return np.nan, np.nan

        # 1. Calculate TRI relative to the start of this specific series slice, starting at 1.0
        tri = (1 + returns_series).cumprod()
        # Get the period *before* the start of the series for the base value
        try:
             freq = pd.infer_freq(returns_series.index) or pd.DateOffset(months=1)
             base_idx = returns_series.index[0] - freq
        except TypeError:
             base_idx = returns_series.index[0] - pd.Timedelta(days=30) # Approximation

        # Use pd.concat instead of .loc to avoid potential SettingWithCopyWarning
        base_row = pd.Series([1.0], index=[base_idx])
        tri = pd.concat([base_row, tri])
        tri.sort_index(inplace=True)


        # 2. Calculate rolling maximum value seen so far (peaks)
        rolling_max = tri.cummax()

        # 3. Calculate drawdown series: (Current Value / Peak Value) - 1
        drawdown = (tri / rolling_max) - 1.0

        # 4. Find the maximum drawdown value (minimum of the drawdown series)
        max_dd = drawdown.min()
        # If no drawdown occurred (or only positive returns), return 0
        if max_dd >= -FLOAT_COMPARISON_THRESHOLD or pd.isnull(max_dd):
             return 0.0, 0.0 # Max Drawdown is 0, Recovery is 0 months

        # 5. Find the peak and trough dates associated with the maximum drawdown
        trough_date = drawdown.idxmin()
        # Find the date of the peak *before* this specific trough
        # Ensure peak_candidates includes the base index
        peak_candidates = rolling_max.loc[:trough_date]

        # Get the first index where the peak value (rolling_max at trough date) was achieved
        peak_value_at_trough = peak_candidates.iloc[-1]
        # Find first occurrence of the peak value before or at the trough
        peak_date = peak_candidates[peak_candidates >= peak_value_at_trough - FLOAT_COMPARISON_THRESHOLD].index[0]
        peak_value = tri.loc[peak_date]


        # 6. Find recovery date and calculate duration
        duration_months: Optional[float] = np.nan # Default to NaN (not recovered)
        try:
            # Look for the first time the TRI reaches or exceeds the peak value *after* the trough date
            recovery_candidates = tri.loc[trough_date:]
            # Find indices where value >= peak_value (within float tolerance)
            recovered_series = recovery_candidates[recovery_candidates >= peak_value - FLOAT_COMPARISON_THRESHOLD]

            if not recovered_series.empty:
                recovery_date = recovered_series.index[0]
                # Calculate duration in months using index positions within the *original returns series*
                # Find integer locations in the original returns_series index
                try:
                     # Get location relative to the start of the returns_series
                     peak_loc_in_returns = returns_series.index.get_loc(peak_date) if peak_date in returns_series.index else -1
                     recovery_loc_in_returns = returns_series.index.get_loc(recovery_date)
                     # Duration is number of periods from peak *start* to recovery *end*
                     # If peak was before returns_series start (base index), start count from 0
                     # Add 1 because duration includes start and end month? No, difference is correct.
                     start_loc = max(0, peak_loc_in_returns) if peak_loc_in_returns != -1 else 0
                     duration_months = float(recovery_loc_in_returns - start_loc)

                except KeyError:
                     # Fallback if dates don't align perfectly, use integer difference in TRI index
                     peak_loc = tri.index.get_loc(peak_date)
                     recovery_loc = tri.index.get_loc(recovery_date)
                     duration_months = float(recovery_loc - peak_loc) # Number of periods in TRI index

        except Exception as e:
            print(f"Warning: Error calculating recovery for series {returns_series.name or ''}: {e}")
            duration_months = np.nan # Ensure NaN on error

        return max_dd, duration_months

    def _calculate_calmar_ratio(self, ann_return: float, max_drawdown: float) -> float:
        """Calculates Calmar Ratio (Annualized Return / Abs(Max Drawdown))."""
        # Handle zero, positive, near-zero, or NaN drawdown
        if max_drawdown is None or max_drawdown >= -FLOAT_COMPARISON_THRESHOLD or abs(max_drawdown) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(max_drawdown):
            return np.nan
        # Avoid division by zero explicitly
        if abs(max_drawdown) < FLOAT_COMPARISON_THRESHOLD:
             return np.nan
        return ann_return / abs(max_drawdown)

    def _calculate_monthly_stats(self, returns_series: pd.Series) -> Tuple[float, float, float]:
        """Calculates % Positive Months, Avg Positive Month Return, Avg Negative Month Return."""
        if returns_series.empty: return np.nan, np.nan, np.nan

        pos_mask = returns_series > FLOAT_COMPARISON_THRESHOLD # Consider near-zero as non-positive
        neg_mask = returns_series < -FLOAT_COMPARISON_THRESHOLD # Consider near-zero as non-negative

        num_months = len(returns_series)
        pct_positive = pos_mask.sum() / num_months if num_months > 0 else np.nan

        # Return 0.0 average if no positive/negative months exist, avoids NaN warning
        avg_positive = returns_series[pos_mask].mean() if pos_mask.any() else 0.0
        avg_negative = returns_series[neg_mask].mean() if neg_mask.any() else 0.0

        # Handle potential NaN results from mean() if masks are empty, though covered above
        avg_positive = 0.0 if pd.isnull(avg_positive) else avg_positive
        avg_negative = 0.0 if pd.isnull(avg_negative) else avg_negative


        return pct_positive, avg_positive, avg_negative

    def analyze_performance(self, periods_years: List[int] = [1, 3, 5, 10]) -> Dict[str, pd.DataFrame]:
        """
        Calculates performance metrics over specified rolling periods and the full sample.

        Args:
            periods_years: List of lookback periods in years (e.g., [1, 3, 5, 10]).

        Returns:
            Dictionary where keys are period labels ('Full Sample', '1Y', etc.)
            and values are formatted DataFrames of performance metrics for that period
            (Index=Strategy/Asset Name, Columns=Metric Name).
        """
        # Dictionary to store final DataFrames for each period
        period_results: Dict[str, pd.DataFrame] = {}
        end_date = self.all_returns.index.max()

        # Define periods to analyze: 'Full Sample' + requested N-Year periods if data allows
        periods_def = {'Full Sample': None} # None indicates using the full common period
        for p_yr in sorted(periods_years):
             # Only include N-Year period if the *total* history is long enough
             if p_yr * 12 <= len(self.all_returns):
                 periods_def[f'{p_yr}Y'] = p_yr * 12 # Store lookback length in months

        print(f"Analyzing periods: {list(periods_def.keys())}")

        # --- Iterate through each defined period ---
        for period_label, period_months in periods_def.items():
            print(f"  Analyzing Period: {period_label}")
            # Dictionary to store results for {strategy: {metric: value}} within this period
            period_data = {}

            # --- Iterate through each strategy/asset ---
            for name in self.all_returns.columns:
                returns_slice = pd.Series(dtype=float) # Initialize empty slice
                valid_slice = False

                if period_months: # N-Year lookback period
                    # Determine start date for the N-year slice ending at the latest date
                    # Use DateOffset for more reliable month calculation
                    start_date_period = end_date - pd.DateOffset(months=period_months -1)

                    # Ensure the slice start isn't before the common analysis start date
                    actual_start_date = max(start_date_period, self.common_start_date)

                    # Get the return data for this strategy and period, drop any NaNs within
                    # Ensure slicing includes the end_date
                    _slice = self.all_returns.loc[actual_start_date:end_date, name].dropna()

                    # Check if sufficient data points exist (e.g., >= 90% of expected months)
                    # Be slightly lenient with required points due to potential date alignment issues
                    min_required_points = max(2, int(period_months * 0.85)) # Require at least 2 points
                    if len(_slice) >= min_required_points:
                        returns_slice = _slice
                        valid_slice = True
                    # else: # Optional warning if skipping specific strategy/period combo
                    #     print(f"      - Skipping {name} for {period_label}: Insufficient data ({len(_slice)} < {min_required_points})")


                else: # 'Full Sample' period (use all common data)
                    returns_slice = self.all_returns[name].dropna()
                    if not returns_slice.empty:
                         valid_slice = True

                # --- Calculate Metrics for Valid Slice ---
                if valid_slice:
                    ann_ret = self._calculate_annualized_return(returns_slice)
                    ann_vol = self._calculate_annualized_volatility(returns_slice)
                    info_ratio = self._calculate_information_ratio(ann_ret, ann_vol)
                    max_dd, recovery_months = self._calculate_max_drawdown(returns_slice)
                    calmar = self._calculate_calmar_ratio(ann_ret, max_dd)
                    pct_pos, avg_pos, avg_neg = self._calculate_monthly_stats(returns_slice)

                    metrics_values = {
                        "Annualized Return": ann_ret, "Annualized Volatility": ann_vol, "Information Ratio": info_ratio,
                        "Max Drawdown": max_dd, "Calmar Ratio": calmar, "Max Drawdown Recovery (Months)": recovery_months,
                        "% Positive Months": pct_pos, "Avg Positive Month Ret": avg_pos, "Avg Negative Month Ret": avg_neg
                    }
                else: # If slice was invalid (e.g., insufficient data)
                    metrics_values = {metric: np.nan for metric in self.metrics}

                period_data[name] = metrics_values # Store results for this strategy

            # --- Assemble and Format Results for the Period ---
            period_summary_df = pd.DataFrame.from_dict(period_data, orient='index')
            # Ensure consistent column order
            period_summary_df = period_summary_df.reindex(columns=self.metrics)

            # Apply formatting for display (Percentages, Decimals, Integers)
            formatted_df = period_summary_df.copy()
            format_pct = ["Annualized Return", "Annualized Volatility", "Max Drawdown",
                          "% Positive Months", "Avg Positive Month Ret", "Avg Negative Month Ret"]
            format_ratio = ["Information Ratio", "Calmar Ratio"]
            format_int = ["Max Drawdown Recovery (Months)"]

            # Define formatting functions for clarity
            def fmt_pct(x): return f"{x:.2%}" if pd.notnull(x) and np.isfinite(x) else "-"
            def fmt_ratio(x): return f"{x:.2f}" if pd.notnull(x) and np.isfinite(x) else "-"
            def fmt_int(x):
                if pd.notnull(x) and np.isfinite(x):
                    return f"{int(x):,d}"
                elif pd.isnull(x):
                    # Check if max_dd was 0, if so, recovery is 0, otherwise ongoing
                    # Need original max_dd value here... This formatting makes assumptions.
                    # Let's adjust: Format NaN as "Ongoing" only if MaxDD was < 0
                    # This requires passing the unformatted df, making it complex here.
                    # Simpler: Treat NaN recovery as "Ongoing" for now.
                    return "Ongoing"
                else: # Handles potential -inf/inf if calculation goes wrong
                    return "-"

            for col in format_pct:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(fmt_pct)
            for col in format_ratio:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(fmt_ratio)
            for col in format_int:
                 if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(fmt_int)


            # Store the formatted DataFrame for this period
            period_results[period_label] = formatted_df

        return period_results


def generate_portfolio_report(portfolio_constructor: PortfolioConstructor,
                              analyzer_periods: List[int] = [1, 3, 5, 10],
                              output_filename: str = "portfolio_report.xlsx") -> None:
    """
    Generates a multi-sheet Excel report containing portfolio analysis results.

    Sheets include:
    1. Portfolio TRI: Total Return Index chart data for constructed portfolios.
    2. Performance Summary: Tables of key metrics for various periods.
    3. Asset Correlation: Correlation matrix of the underlying assets.
    4. Weights Sheets: Separate sheet for weights of each constructed portfolio.

    Args:
        portfolio_constructor: An initialized PortfolioConstructor instance after
                               running construction methods.
        analyzer_periods: List of periods (in years) for performance analysis tables.
        output_filename: The desired name for the output Excel file.
    """
    print(f"\nGenerating full Excel report: {output_filename}...")

    # --- Retrieve Data from Constructor ---
    portfolio_tri_df = portfolio_constructor.get_total_return_indexes()
    portfolio_returns_df = portfolio_constructor.get_returns()
    asset_returns_df = portfolio_constructor.returns_df # Underlying asset returns
    all_weights_dict = portfolio_constructor.weights

    # Basic check if there's any data to report
    if portfolio_tri_df.empty and not all_weights_dict and asset_returns_df.empty:
        print("No portfolio TRI, weights, or asset returns found. Report generation skipped.")
        return

    # --- Perform Performance Analysis ---
    analyzer = None
    summary_tables_dict: Dict[str, pd.DataFrame] = {}
    # Ensure both portfolio and asset returns are available for analysis
    if not portfolio_returns_df.empty and not asset_returns_df.empty:
         try:
             # Instantiate and run analyzer
             analyzer = PortfolioAnalyzer(portfolio_returns_df, asset_returns_df)
             summary_tables_dict = analyzer.analyze_performance(periods_years=analyzer_periods)
         except ValueError as e: print(f"Could not initialize/run PortfolioAnalyzer: {e}")
         except Exception as e: print(f"Unexpected error during performance analysis: {e}")
    else: print("Skipping performance analysis: Portfolio or Asset returns missing.")


    # --- Calculate Asset Correlation ---
    asset_correlation_matrix = pd.DataFrame()
    if not asset_returns_df.empty:
        try:
             # Use the full history of asset returns available from the constructor
             asset_correlation_matrix = asset_returns_df.corr().round(4)
             print("Calculated asset correlation matrix.")
        except Exception as e: print(f"Warning: Could not calculate asset correlation matrix: {e}")

    # --- Write All Data to Excel ---
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            current_sheet_num = 1 # Counter for feedback

            # --- Sheet 1: Portfolio TRI ---
            print(f"({current_sheet_num}) Writing Portfolio TRI sheet...")
            if not portfolio_tri_df.empty:
                portfolio_tri_df.to_excel(writer, sheet_name="Portfolio TRI", index=True, index_label="Date")
                print(f"    - Done ({len(portfolio_tri_df)} rows)")
            else: print("    - Skipped (no data).")
            current_sheet_num += 1

            # --- Sheet 2: Performance Summary (Multiple Tables) ---
            summary_sheet_name = "Performance Summary"
            print(f"({current_sheet_num}) Writing {summary_sheet_name} sheet...")
            if analyzer and summary_tables_dict:
                current_row = 0
                # Define desired order for tables
                period_order = ['Full Sample'] + [f'{p}Y' for p in sorted(analyzer_periods)]
                # Write each period's table sequentially down the sheet
                for period_label in period_order:
                    if period_label in summary_tables_dict:
                        df_to_write = summary_tables_dict[period_label]
                        if not df_to_write.empty:
                             # Write Title Row (Merged cells might require openpyxl directly later)
                             title_df = pd.DataFrame([f"{period_label} Performance Summary"])
                             title_df.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=False, header=False)
                             current_row += 1 # Move below title
                             # Write Data Table
                             df_to_write.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=True, header=True)
                             # Update row pointer for next table (+ header + data rows + 2 blank rows)
                             current_row += (len(df_to_write) + 3)
                             print(f"    - Added table for: {period_label}")
                        # else: print(f"    - Skipping empty table for: {period_label}")
                    # else: print(f"    - No results for period: {period_label}")
                print(f"    - Done")
            else: print(f"    - Skipped (no analysis results).")
            current_sheet_num += 1

            # --- Sheet 3: Asset Correlation Matrix ---
            corr_sheet_name = "Asset Correlation"
            print(f"({current_sheet_num}) Writing {corr_sheet_name} sheet...")
            if not asset_correlation_matrix.empty:
                 asset_correlation_matrix.to_excel(writer, sheet_name=corr_sheet_name, index=True, header=True)
                 print(f"    - Done ({len(asset_correlation_matrix)} rows)")
            else: print(f"    - Skipped (no data or calculation failed).")
            current_sheet_num +=1

            # --- Subsequent Sheets: Weights ---
            print(f"({current_sheet_num}+) Writing Weight sheets...")
            if not all_weights_dict: print("    - Skipped (no weights calculated).")
            else:
                sheet_count = 0
                for method_name, weights_df in all_weights_dict.items():
                    # Sanitize sheet name for Excel (max 31 chars, no invalid chars)
                    safe_sheet_name = "".join(c for c in method_name if c.isalnum() or c in (' ', '_', '-'))[:30] # Limit length
                    sheet_name = f"{safe_sheet_name} Weights"

                    if not weights_df.empty:
                        weights_df.to_excel(writer, sheet_name=sheet_name, index=True, index_label="Date")
                        print(f"    - Writing sheet: {sheet_name} ({len(weights_df)} rows)")
                        sheet_count += 1
                    else: print(f"    - Skipping {sheet_name} (no data).")
                print(f"    - Done ({sheet_count} weight sheets written)")
                current_sheet_num += sheet_count


        print(f"\nExcel report '{output_filename}' generated successfully.")

    except ImportError: print("\nError: 'openpyxl' required for Excel writing. Please install it (`pip install openpyxl`).");
    except Exception as e: print(f"\nAn error occurred while writing the Excel file: {e}")


# --- Example Usage Script ---
if __name__ == "__main__":
    """
    Example script demonstrating how to use the PortfolioConstructor and
    PortfolioAnalyzer classes and generate the Excel report.
    """

    # --- 1. Simulation Setup ---
    print("--- Setting up Simulation ---")
    np.random.seed(42) # For reproducible random results
    # Define simulation period (e.g., 10 years of monthly data)
    dates = pd.date_range(start='2014-01-31', periods=120, freq='M') # Use Month End frequency
    n_assets = 5 # Increased assets slightly
    tickers = [f'Asset_{chr(65+i)}' for i in range(n_assets)] # Example tickers

    # Simulate monthly returns for assets
    sim_returns_data = {
        tickers[0]: np.random.normal(0.008, 0.04, len(dates)), # Lower return/vol
        tickers[1]: np.random.normal(0.012, 0.06, len(dates)), # Higher return/vol
        tickers[2]: np.random.normal(0.010, 0.05, len(dates)), # Medium return/vol
        tickers[3]: np.random.normal(0.005, 0.03, len(dates)), # Lowest return/vol
        tickers[4]: np.random.normal(0.009, 0.045, len(dates)) # Another medium one
    }
    # Introduce some correlation
    sim_returns_data[tickers[1]] += 0.3 * sim_returns_data[tickers[2]] # Correlate B and C
    sim_returns_data[tickers[4]] += 0.2 * sim_returns_data[tickers[0]] # Correlate E and A

    sim_returns = pd.DataFrame(sim_returns_data, index=dates)


    # Create a Total Return Index (TRI) from simulated returns, starting at 100
    sim_tri = (1 + sim_returns).cumprod() * 100
    # Add the base value row at T-1
    start_date_tri = sim_tri.index.min() - pd.DateOffset(months=1) # Use DateOffset
    sim_tri.loc[start_date_tri] = 100.0
    sim_tri.sort_index(inplace=True)

    print("Sample Total Return Index Head:")
    print(sim_tri.head())
    print("-" * 30)

    # --- 2. Portfolio Construction and Analysis ---
    try:
        print("\n--- Running Portfolio Construction ---")
        # Initialize the constructor with the TRI data
        constructor = PortfolioConstructor(tri_df=sim_tri)

        # --- Run different portfolio construction methods ---

        # Monthly Rebalancing (Default)
        constructor.construct_equal_weight() # EW is always monthly implicitly
        constructor.construct_equal_volatility(lookback_years=3, rebalance_freq_months=1)
        constructor.construct_erc(lookback_years=3, use_ewm=True, rebalance_freq_months=1)
        constructor.construct_erc(lookback_years=3, use_ewm=False, rebalance_freq_months=1)
        constructor.construct_hrp(lookback_years=3, linkage_method='ward', rebalance_freq_months=1)
        constructor.construct_hrp(lookback_years=3, linkage_method='single', rebalance_freq_months=1)

        # Annual Rebalancing (Recalculate weights every 12 months)
        constructor.construct_equal_volatility(lookback_years=3, rebalance_freq_months=12)
        constructor.construct_erc(lookback_years=3, use_ewm=True, rebalance_freq_months=12)
        constructor.construct_erc(lookback_years=3, use_ewm=False, rebalance_freq_months=12)
        constructor.construct_hrp(lookback_years=3, linkage_method='ward', rebalance_freq_months=12)
        constructor.construct_hrp(lookback_years=3, linkage_method='single', rebalance_freq_months=12)


        print("\n--- Generating Final Report ---")
        # Define analysis periods and output filename
        analysis_periods_config = [1, 3, 5, 10]
        output_file = "portfolio_analysis_report_rebal_freq.xlsx"

        # Generate the multi-sheet Excel report
        generate_portfolio_report(
            constructor,
            analyzer_periods=analysis_periods_config,
            output_filename=output_file
        )

    except ValueError as ve:
         # Handle specific known errors (e.g., bad input data)
         print(f"\nERROR: A ValueError occurred during processing: {ve}")
    except KeyError as ke:
         # Handle errors related to missing dictionary keys (e.g., incorrect method name)
         print(f"\nERROR: A KeyError occurred: {ke}")
    except ImportError as ie:
         # Handle missing optional dependency (openpyxl)
         print(f"\nERROR: Missing required library: {ie}. Please install 'openpyxl'.")
    except Exception as e:
         # Catch any other unexpected errors
         print(f"\nERROR: An unexpected error occurred: {e}")
         import traceback # Print detailed traceback for debugging
         traceback.print_exc()
         # Optionally, re-raise the exception if debugging is needed
         # raise e

