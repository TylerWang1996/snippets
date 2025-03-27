import pandas as pd
import numpy as np
from scipy.optimize import minimize
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
# Small value added to the diagonal of non-Positive Semi-Definite covariance matrices
COV_MATRIX_JITTER = 1e-8

# --- Warning Suppression ---
# Suppress common warnings during optimization or sqrt calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress FutureWarnings often arising from library interactions
warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioConstructor:
    """
    Constructs portfolios using Equal Weight, Equal Volatility, and Equal
    Risk Contribution methods based on monthly Total Return Index (TRI) data.

    Calculates weights and resulting monthly portfolio returns, handling lookahead bias.

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
        self.returns_df = _tri_df.pct_change().dropna(axis=0, how='any')
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

        Results are stored in `self.weights` and `self.returns`.

        Args:
            method_name: Name of the portfolio construction method (e.g., 'EW').
            weights_df: DataFrame of portfolio weights, indexed by rebalance dates.
                        These are the target weights decided *at* the index date.
        """
        if weights_df.empty:
             print(f"Warning: Received empty weights DataFrame for '{method_name}'. Cannot calculate returns.")
             self.weights[method_name] = pd.DataFrame(columns=self.tickers)
             self.returns[method_name] = pd.Series(dtype=float, name=method_name)
             return

        # Align weights to full return series index, forward filling between rebalances
        aligned_weights = weights_df.reindex(self.returns_df.index).ffill()
        # Normalize weights (accounts for potential float issues / ensures sum=1)
        aligned_weights = aligned_weights.div(aligned_weights.sum(axis=1) + FLOAT_COMPARISON_THRESHOLD, axis=0)

        # --- Crucial step to avoid lookahead bias ---
        # Shift weights by 1 period: weights decided using data up to time 't'
        # (index date) are applied to returns from 't' to 't+1'.
        shifted_weights = aligned_weights.shift(1)

        # Calculate portfolio returns: Sum of (weight * asset return)
        # Uses element-wise multiplication then sums across assets (axis=1)
        portfolio_returns = (shifted_weights * self.returns_df).sum(axis=1)

        # Filter results to start only when weights were first available
        first_valid_weight_date = weights_df.index.min()
        portfolio_returns = portfolio_returns.loc[first_valid_weight_date:]

        # Store the actual weights *used* for the calculation period (i.e., the shifted weights)
        effective_weights = shifted_weights.loc[first_valid_weight_date:].dropna(axis=0, how='all')
        self.weights[method_name] = effective_weights

        # Store the calculated portfolio returns (dropna removes initial NaN from shift)
        self.returns[method_name] = portfolio_returns.dropna().rename(method_name)

        # User feedback on results
        if not self.returns[method_name].empty:
            print(f"Calculated returns for {method_name} from {self.returns[method_name].index.min().strftime('%Y-%m-%d')} to {self.returns[method_name].index.max().strftime('%Y-%m-%d')}")
        else:
            print(f"No returns were calculated for {method_name} (check data length vs lookback).")

    def construct_equal_weight(self) -> None:
        """Constructs an Equal Weight (EW) portfolio (1/N weight per asset)."""
        method_name = "EW"
        print(f"\nConstructing {method_name} portfolio...")
        if self.num_assets == 0:
             print(f"Warning: Cannot construct {method_name} with zero assets."); return

        # Calculate equal weight value
        weight_value = 1.0 / self.num_assets
        weights = pd.Series(weight_value, index=self.tickers)

        # Create DataFrame of weights matching the returns index for calculation
        weights_array = np.tile(weights.values, (len(self.returns_df.index), 1))
        weights_df = pd.DataFrame(weights_array, index=self.returns_df.index, columns=self.tickers)

        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")

    def construct_equal_volatility(self, lookback_years: int = 2) -> None:
        """
        Constructs an Equal Volatility (EV) portfolio. Weights are inversely
        proportional to each asset's rolling annualized volatility.

        Args:
            lookback_years: Lookback period in years for volatility calculation.
        """
        method_name = "EV"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years)...")

        # Validate lookback period
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
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

        # Iterate through dates where lookback is available
        for date in self.returns_df.loc[first_calc_date:].index:
            # Get data slice for lookback ending at current date 't'
            current_loc = self.returns_df.index.get_loc(date)
            start_loc = current_loc - lookback_window + 1
            returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]

            # Calculate annualized volatility (std dev * sqrt(12))
            rolling_vol = returns_slice.std() * ANNUALIZATION_FACTOR_VOL

            # Handle near-zero volatility to avoid division by zero
            rolling_vol[rolling_vol.abs() < SMALL_VOL_THRESHOLD] = SMALL_VOL_THRESHOLD

            # Calculate inverse volatility weights: w_i = (1/vol_i) / sum(1/vol_j)
            inv_vol = 1.0 / rolling_vol
            weights = inv_vol / inv_vol.sum()
            # Re-normalize for safety (handles potential float precision issues)
            weights = weights / weights.sum()
            weights_dict[date] = weights

        if not weights_dict:
             print(f"Warning: No weights were calculated for {method_name}."); return # Should not happen if initial check passes

        # Create DataFrame and calculate portfolio returns
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers)
        self._calculate_portfolio_returns(method_name, weights_df)
        print(f"{method_name} construction complete.")

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

        Args:
            cov_matrix: The annualized covariance matrix (pd.DataFrame).
            calculation_date: The date for which weights are being calculated (for logging).

        Returns:
            pd.Series containing the optimized weights, indexed by ticker.
            Returns Equal Weights if optimization fails.
        """
        # --- Input Validation and Alignment ---
        if cov_matrix.shape != (self.num_assets, self.num_assets):
             raise ValueError(f"[{calculation_date.date()}] Cov matrix shape mismatch.")
        # Ensure matrix columns/index match self.tickers order
        if not np.all(cov_matrix.columns == self.tickers) or not np.all(cov_matrix.index == self.tickers):
             cov_matrix = cov_matrix.reindex(index=self.tickers, columns=self.tickers)
             if cov_matrix.isnull().values.any():
                  raise ValueError(f"[{calculation_date.date()}] Cov matrix has NaNs after reindexing.")

        cov_matrix_values = cov_matrix.values # Use numpy array for optimization

        # --- PSD Check & Jitter ---
        # Ensure matrix is Positive Semi-Definite for optimizer stability
        try:
            np.linalg.cholesky(cov_matrix_values)
        except np.linalg.LinAlgError:
            # Add small diagonal jitter if not PSD
            cov_matrix_values += np.eye(self.num_assets) * COV_MATRIX_JITTER

        # --- Optimization Setup ---
        # Initial guess: Equal weights
        w0 = np.ones(self.num_assets) / self.num_assets
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # Bounds: Weights between 0 and 1 (long-only)
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        # --- Run Optimizer ---
        result = minimize(
            self._erc_objective,
            w0,
            args=(cov_matrix_values,),
            method='SLSQP', # Suitable for bounds and equality constraints
            bounds=bounds,
            constraints=constraints,
            tol=OPTIMIZER_TOLERANCE,
            options={'maxiter': 1000, 'ftol': OPTIMIZER_TOLERANCE} # Also use ftol
        )

        # --- Handle Results ---
        if not result.success:
            # Fallback to Equal Weight if optimization fails
            print(f"Warning [{calculation_date.date()}]: ERC optimization failed: {result.message}. Falling back to EW.")
            return pd.Series(w0, index=self.tickers)

        # Normalize final weights for precision and return as Series
        optimized_weights = result.x / result.x.sum()
        return pd.Series(optimized_weights, index=self.tickers)

    def construct_erc(self, lookback_years: int = 2, use_ewm: bool = True) -> None:
        """
        Constructs an Equal Risk Contribution (ERC) portfolio using optimization.

        Args:
            lookback_years: Lookback period in years for covariance calculation.
            use_ewm: If True, use Exponentially Weighted Moving Covariance;
                     otherwise, use Simple Moving Covariance.
        """
        method_name = f"ERC_{'EWM' if use_ewm else 'Simple'}"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years)...")

        # Validate lookback period
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        lookback_window = int(lookback_years * 12)
        min_lookback_required = 2
        if lookback_window < min_lookback_required: raise ValueError(f"Lookback window must be >= {min_lookback_required} months.")
        if lookback_window > len(self.returns_df): raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")

        weights_dict = {}
        # Determine first date calculation is possible
        first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index):
            print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return

        first_calc_date = self.returns_df.index[first_calc_date_index]

        # Iterate through dates where lookback is available
        for date in self.returns_df.loc[first_calc_date:].index:
            # Get data slice for lookback ending at current date 't'
            current_loc = self.returns_df.index.get_loc(date)
            start_loc = current_loc - lookback_window + 1
            returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]

            cov_matrix = None
            cov_calc_successful = False
            # Calculate annualized covariance matrix (Simple or EWM)
            try:
                if use_ewm:
                    # EWMA covariance calculation
                    ewm_cov = returns_slice.ewm(
                        span=lookback_window,
                        min_periods=lookback_window, # Ensures full window for first calc
                        adjust=True
                    ).cov(pairwise=True) * ANNUALIZATION_FACTOR_COV
                    # Extract matrix for the specific date from MultiIndex result
                    if date in ewm_cov.index.get_level_values(0):
                         cov_matrix = ewm_cov.loc[date]
                else:
                    # Simple moving covariance calculation
                    cov_matrix = returns_slice.cov() * ANNUALIZATION_FACTOR_COV

                # Validate calculated matrix
                if isinstance(cov_matrix, pd.DataFrame) and not cov_matrix.isnull().values.any() and cov_matrix.shape == (self.num_assets, self.num_assets):
                    cov_calc_successful = True
                else:
                     # Only print warning if matrix calculation yielded invalid result
                     if cov_matrix is not None:
                          print(f"Warning [{date.date()}]: Invalid cov matrix ({'EWM' if use_ewm else 'Simple'}). Skipping.")

            except Exception as e:
                print(f"Error [{date.date()}] calculating cov matrix: {e}. Skipping.")

            # Calculate weights only if covariance matrix is valid
            if cov_calc_successful:
                try:
                    erc_weights = self._calculate_erc_weights(cov_matrix, date)
                    weights_dict[date] = erc_weights
                except Exception as e:
                     print(f"Error [{date.date()}] calculating ERC weights: {e}. Skipping.")

        if not weights_dict:
             print(f"Warning: No weights calculated for {method_name}."); return

        # Create DataFrame and calculate portfolio returns
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers).fillna(0) # Ensure all columns exist
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
            method_name: The name of the method (e.g., 'EW', 'ERC_EWM').

        Returns:
            DataFrame of weights (index=Date, columns=Tickers).

        Raises:
            KeyError: If the method name is not found.
        """
        if method_name not in self.weights:
            available = list(self.weights.keys())
            raise KeyError(f"Weights for '{method_name}' not found. Available: {available}")
        return self.weights[method_name]

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
            common_start_date = valid_returns.dropna().index.min()
        except ValueError: # Handles case where dropna() makes the DataFrame empty
            print("Cannot generate TRI: No common start date found across portfolios."); return pd.DataFrame()

        # Filter returns from the common start date
        common_returns = valid_returns.loc[common_start_date:]
        if common_returns.empty: print(f"Cannot generate TRI: No common returns data from {common_start_date}."); return pd.DataFrame()

        # Calculate cumulative product and scale to base value
        tri = (1 + common_returns).cumprod() * base_value

        # --- Prepend Base Value Row ---
        # Create a row with the base value dated one day before the common start date
        base_date = common_start_date - pd.Timedelta(days=1)
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

        # Combine all returns; drop strategies/assets with no data at all
        self.all_returns = pd.concat([portfolio_returns, asset_returns], axis=1)
        self.all_returns.dropna(axis=1, how='all', inplace=True)
        if self.all_returns.empty:
            raise ValueError("Combined returns are empty after dropping NaN columns.")

        # Determine common analysis period (where all included series have data)
        try:
            # Find first/last date where *at least one* series has data
            start_dt = self.all_returns.dropna(how='all').index.min()
            end_dt = self.all_returns.dropna(how='all').index.max()
            # Slice to this maximal range first
            self.all_returns = self.all_returns.loc[start_dt:end_dt]
            # Now find the common start based on non-NaN values within this range
            self.common_start_date = self.all_returns.dropna().index.min()
            self.common_end_date = end_dt # Use the already found end date
            # Final slice to the strictly common period
            self.all_returns = self.all_returns.loc[self.common_start_date:self.common_end_date]
            # Final check if slicing resulted in empty df (e.g., only one common date)
            if self.all_returns.dropna().empty: raise ValueError("No overlapping non-NaN data.")
        except Exception as e:
             raise ValueError(f"Error determining common date range or slicing returns: {e}")

        print(f"\nInitializing Analyzer. Common analysis period: {self.common_start_date.strftime('%Y-%m-%d')} to {self.common_end_date.strftime('%Y-%m-%d')}")

        # Define the metrics to be calculated
        self.metrics = [
            "Annualized Return", "Annualized Volatility", "Information Ratio",
            "Max Drawdown", "Calmar Ratio", "Max Drawdown Recovery (Months)",
            "% Positive Months", "Avg Positive Month Ret", "Avg Negative Month Ret"
        ]

    def _calculate_annualized_return(self, returns_series: pd.Series) -> float:
        """Calculates geometric annualized return from monthly returns."""
        if returns_series.empty or len(returns_series) == 0: return np.nan
        num_years = len(returns_series) / 12.0
        if num_years <= 0: return np.nan
        # Handle returns <= -100%, which invalidate geometric linking
        if ((1 + returns_series) <= 0).any(): return np.nan
        # Geometric mean formula: (product(1+r)^(1/num_years)) - 1
        return ((1 + returns_series).prod())**(1 / num_years) - 1

    def _calculate_annualized_volatility(self, returns_series: pd.Series) -> float:
        """Calculates annualized volatility (std dev) from monthly returns."""
        # Need at least 2 data points to calculate std dev
        if returns_series.empty or len(returns_series) < 2: return np.nan
        return returns_series.std() * ANNUALIZATION_FACTOR_VOL

    def _calculate_information_ratio(self, ann_return: float, ann_vol: float) -> float:
        """
        Calculates Information Ratio (Annualized Return / Annualized Volatility).
        Note: This is sometimes called Sharpe Ratio when risk-free rate is zero.
        """
        # Handle zero, near-zero, or NaN volatility
        if ann_vol is None or abs(ann_vol) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(ann_vol):
            return np.nan
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
        base_idx = returns_series.index[0] - pd.Timedelta(days=1) # T-1 index for base
        tri.loc[base_idx] = 1.0
        tri.sort_index(inplace=True)

        # 2. Calculate rolling maximum value seen so far (peaks)
        rolling_max = tri.cummax()

        # 3. Calculate drawdown series: (Current Value / Peak Value) - 1
        drawdown = (tri / rolling_max) - 1.0

        # 4. Find the maximum drawdown value (minimum of the drawdown series)
        max_dd = drawdown.min()
        # If no drawdown occurred (or only positive returns), return 0
        if max_dd >= -FLOAT_COMPARISON_THRESHOLD or pd.isnull(max_dd):
             return 0.0, 0.0

        # 5. Find the peak and trough dates associated with the maximum drawdown
        trough_date = drawdown.idxmin()
        # Find the date of the peak *before* this specific trough
        peak_candidates = rolling_max.loc[:trough_date]
        # Get the first index where the peak value was achieved before the trough
        peak_date = peak_candidates[peak_candidates == peak_candidates.iloc[-1]].index[0]
        peak_value = tri.loc[peak_date]

        # 6. Find recovery date and calculate duration
        duration_months: Optional[float] = np.nan # Default to NaN (not recovered)
        try:
            # Look for the first time the TRI reaches or exceeds the peak value *after* the trough date
            recovery_candidates = tri.loc[trough_date:]
            recovered_series = recovery_candidates[recovery_candidates >= peak_value - FLOAT_COMPARISON_THRESHOLD]

            if not recovered_series.empty:
                 recovery_date = recovered_series.index[0]
                 # Calculate duration in months using index positions within the TRI series (includes base)
                 peak_loc = tri.index.get_loc(peak_date)
                 recovery_loc = tri.index.get_loc(recovery_date)
                 duration_months = float(recovery_loc - peak_loc) # Periods between peak and recovery

        except Exception as e:
             print(f"Warning: Error calculating recovery for series {returns_series.name or ''}: {e}")
             duration_months = np.nan # Ensure NaN on error

        return max_dd, duration_months

    def _calculate_calmar_ratio(self, ann_return: float, max_drawdown: float) -> float:
        """Calculates Calmar Ratio (Annualized Return / Abs(Max Drawdown))."""
        # Handle zero, positive, near-zero, or NaN drawdown
        if max_drawdown is None or max_drawdown >= -FLOAT_COMPARISON_THRESHOLD or abs(max_drawdown) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(max_drawdown):
            return np.nan
        return ann_return / abs(max_drawdown)

    def _calculate_monthly_stats(self, returns_series: pd.Series) -> Tuple[float, float, float]:
        """Calculates % Positive Months, Avg Positive Month Return, Avg Negative Month Return."""
        if returns_series.empty: return np.nan, np.nan, np.nan

        pos_mask = returns_series > 0
        neg_mask = returns_series < 0

        pct_positive = pos_mask.mean() if len(returns_series) > 0 else np.nan
        # Return 0.0 average if no positive/negative months exist, avoids NaN
        avg_positive = returns_series[pos_mask].mean() if pos_mask.any() else 0.0
        avg_negative = returns_series[neg_mask].mean() if neg_mask.any() else 0.0

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
                    start_date_period = end_date - pd.DateOffset(months=period_months - 1)
                    # Ensure the slice start isn't before the common analysis start date
                    actual_start_date = max(start_date_period, self.common_start_date)
                    # Get the return data for this strategy and period, drop any NaNs within
                    _slice = self.all_returns.loc[actual_start_date:, name].dropna()

                    # Check if sufficient data points exist (e.g., >= 90% of expected months)
                    min_required_points = int(period_months * 0.9)
                    if len(_slice) >= min_required_points:
                        returns_slice = _slice
                        valid_slice = True
                    # else: # Optional warning if skipping specific strategy/period combo
                        # print(f"    - Skipping {name} for {period_label}: Insufficient data ({len(_slice)} < {min_required_points})")

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

            for col in format_pct:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
            for col in format_ratio:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
            for col in format_int:
                 if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x):,d}" if pd.notnull(x) and np.isfinite(x) else ("-" if pd.isnull(x) or not np.isfinite(x) else "Ongoing"))

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
    if not portfolio_returns_df.empty or not asset_returns_df.empty:
         try:
              # Align returns data for the analyzer
              common_idx = portfolio_returns_df.index.union(asset_returns_df.index).sort_values()
              port_ret_aligned = portfolio_returns_df.reindex(common_idx)
              asset_ret_aligned = asset_returns_df.reindex(common_idx)
              # Instantiate and run analyzer
              analyzer = PortfolioAnalyzer(port_ret_aligned, asset_ret_aligned)
              summary_tables_dict = analyzer.analyze_performance(periods_years=analyzer_periods)
         except ValueError as e: print(f"Could not initialize/run PortfolioAnalyzer: {e}")
         except Exception as e: print(f"Unexpected error during performance analysis: {e}")
    else: print("Skipping performance analysis: No returns available.")

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
                portfolio_tri_df.to_excel(writer, sheet_name="Portfolio TRI", index=True)
                print(f"   - Done ({len(portfolio_tri_df)} rows)")
            else: print("   - Skipped (no data).")
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
                             # Write Title Row
                             title_df = pd.DataFrame([f"{period_label} Performance Summary"])
                             title_df.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=False, header=False)
                             current_row += 1 # Move below title
                             # Write Data Table
                             df_to_write.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=True, header=True)
                             # Update row pointer for next table (+ header + data rows + 2 blank rows)
                             current_row += (len(df_to_write) + 3)
                             print(f"   - Added table for: {period_label}")
                        # else: print(f"   - Skipping empty table for: {period_label}") # Optional verbose skip message
                    # else: print(f"   - No results for period: {period_label}") # Optional verbose skip message
                print(f"   - Done")
            else: print(f"   - Skipped (no analysis results).")
            current_sheet_num += 1

            # --- Sheet 3: Asset Correlation Matrix ---
            corr_sheet_name = "Asset Correlation"
            print(f"({current_sheet_num}) Writing {corr_sheet_name} sheet...")
            if not asset_correlation_matrix.empty:
                 asset_correlation_matrix.to_excel(writer, sheet_name=corr_sheet_name, index=True, header=True)
                 print(f"   - Done ({len(asset_correlation_matrix)} rows)")
            else: print(f"   - Skipped (no data or calculation failed).")
            current_sheet_num +=1

            # --- Subsequent Sheets: Weights ---
            print(f"({current_sheet_num}+) Writing Weight sheets...")
            if not all_weights_dict: print("   - Skipped (no weights calculated).")
            else:
                for method_name, weights_df in all_weights_dict.items():
                    sheet_name = f"{method_name} Weights"
                    if not weights_df.empty:
                        weights_df.to_excel(writer, sheet_name=sheet_name, index=True)
                        print(f"   - Writing sheet: {sheet_name} ({len(weights_df)} rows)")
                    else: print(f"   - Skipping {sheet_name} (no data).")
                    current_sheet_num +=1

        print(f"\nExcel report '{output_filename}' generated successfully.")

    except ImportError: print("\nError: 'openpyxl' required. Please install it (`pip install openpyxl`).");
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
    dates = pd.date_range(start='2015-01-31', periods=120, freq='M') # Use Month End frequency
    n_assets = 4
    tickers = [f'Strategy_{chr(65+i)}' for i in range(n_assets)] # Example tickers

    # Simulate monthly returns for assets
    sim_returns = pd.DataFrame({
        tickers[0]: np.random.normal(0.008, 0.04, len(dates)), # Lower return/vol
        tickers[1]: np.random.normal(0.012, 0.06, len(dates)), # Higher return/vol
        tickers[2]: np.random.normal(0.010, 0.05, len(dates)), # Medium return/vol
        tickers[3]: np.random.normal(0.005, 0.03, len(dates)), # Lowest return/vol
    }, index=dates)

    # Create a Total Return Index (TRI) from simulated returns, starting at 100
    sim_tri = (1 + sim_returns).cumprod() * 100
    # Add the base value row at T-1
    start_date_tri = sim_tri.index.min() - pd.DateOffset(months=1)
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

        # Run different portfolio construction methods
        constructor.construct_equal_weight()
        constructor.construct_equal_volatility(lookback_years=2)
        constructor.construct_erc(lookback_years=2, use_ewm=True) # 2yr EWM ERC
        constructor.construct_erc(lookback_years=3, use_ewm=False) # 3yr Simple ERC

        print("\n--- Generating Final Report ---")
        # Define analysis periods and output filename
        analysis_periods_config = [1, 3, 5, 10]
        output_file = "portfolio_analysis_report.xlsx"

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
         # Optionally, re-raise the exception if debugging is needed
         # raise e