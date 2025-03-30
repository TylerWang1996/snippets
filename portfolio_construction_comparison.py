import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
from typing import Dict, List, Tuple, Optional

# --- Constants for Calculations and Thresholds ---
ANNUALIZATION_FACTOR_VOL = np.sqrt(12)
ANNUALIZATION_FACTOR_COV = 12
SMALL_VOL_THRESHOLD = 1e-10
FLOAT_COMPARISON_THRESHOLD = 1e-15
OPTIMIZER_TOLERANCE = 1e-9
COV_MATRIX_JITTER = 1e-8

# --- Warning Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioConstructor:
    """
    Constructs portfolios using Equal Weight, Equal Volatility, and Equal
    Risk Contribution methods based on monthly Total Return Index (TRI) data.

    Calculates weights and resulting monthly portfolio returns, handling lookahead bias.

    Attributes:
        returns_df: DataFrame of monthly returns for the underlying assets/strategies.
        tickers: List of asset/strategy ticker strings used.
        num_assets: Integer count of the assets/strategies used.
        weights: Dict storing calculated portfolio weights {method_name: DataFrame}.
                 Weights represent holdings at the *start* of the month.
        returns: Dict storing calculated monthly returns {method_name: Series}.
    """
    def __init__(self, tri_df: pd.DataFrame) -> None:
        """
        Initializes the constructor, validates input, and calculates asset returns.

        Args:
            tri_df: DataFrame with DatetimeIndex (monthly) and columns for each
                    asset's/strategy's Total Return Index. Should contain only
                    tickers to be included.

        Raises:
            ValueError: If input is invalid or processing yields empty DataFrame.
        """
        if not isinstance(tri_df.index, pd.DatetimeIndex): raise ValueError("Input DataFrame must have a DatetimeIndex.")
        if tri_df.empty: raise ValueError("Input tri_df DataFrame is empty.")
        _tri_df = tri_df.copy().dropna(axis=0, how='all').dropna(axis=1, how='all')
        if _tri_df.empty: raise ValueError("Input DataFrame empty after removing all-NaN rows/columns.")
        _tri_df.sort_index(inplace=True)
        self.returns_df = _tri_df.pct_change().dropna(axis=0, how='any')
        if self.returns_df.empty: raise ValueError("Return calculation resulted in empty DataFrame. Ensure >1 consecutive valid periods per asset.")
        self.tickers = self.returns_df.columns.tolist(); self.num_assets = len(self.tickers)
        self.weights: Dict[str, pd.DataFrame] = {}; self.returns: Dict[str, pd.Series] = {}
        print(f"Initialized PortfolioConstructor with {self.num_assets} assets/strategies: {', '.join(self.tickers)}")
        if not self.returns_df.empty: print(f"Asset return data available from {self.returns_df.index.min().strftime('%Y-%m-%d')} to {self.returns_df.index.max().strftime('%Y-%m-%d')}")

    def _calculate_portfolio_returns(self, method_name: str, weights_df: pd.DataFrame) -> None:
        """Internal: Calculates monthly portfolio returns, avoiding lookahead bias."""
        if weights_df.empty: print(f"Warning: Empty weights for '{method_name}'."); self.weights[method_name]=pd.DataFrame(columns=self.tickers); self.returns[method_name]=pd.Series(dtype=float, name=method_name); return
        aligned_weights = weights_df.reindex(self.returns_df.index).ffill(); aligned_weights = aligned_weights.div(aligned_weights.sum(axis=1) + FLOAT_COMPARISON_THRESHOLD, axis=0)
        shifted_weights = aligned_weights.shift(1) # Avoids lookahead
        portfolio_returns = (shifted_weights * self.returns_df).sum(axis=1)
        try: first_valid_weight_date = weights_df.index.min()
        except ValueError: print(f"Warning: Could not get min date from weights for {method_name}."); return # Handle empty index case
        portfolio_returns = portfolio_returns.loc[first_valid_weight_date:] # Start returns when weights available
        effective_weights = shifted_weights.loc[first_valid_weight_date:].dropna(axis=0, how='all'); self.weights[method_name] = effective_weights
        self.returns[method_name] = portfolio_returns.dropna().rename(method_name)
        if not self.returns[method_name].empty: print(f"Calculated returns for {method_name} from {self.returns[method_name].index.min().strftime('%Y-%m-%d')} to {self.returns[method_name].index.max().strftime('%Y-%m-%d')}")
        else: print(f"No returns calculated for {method_name}.")

    def construct_equal_weight(self) -> None:
        """Constructs an Equal Weight (EW) portfolio."""
        method_name = "EW"; print(f"\nConstructing {method_name} portfolio...")
        if self.num_assets == 0: print(f"Warning: Cannot construct {method_name} with zero assets."); return
        weight_value = 1.0 / self.num_assets; weights = pd.Series(weight_value, index=self.tickers)
        weights_array = np.tile(weights.values, (len(self.returns_df.index), 1)); weights_df = pd.DataFrame(weights_array, index=self.returns_df.index, columns=self.tickers)
        self._calculate_portfolio_returns(method_name, weights_df); print(f"{method_name} construction complete.")

    def construct_equal_volatility(self, lookback_years: int = 2) -> None:
        """Constructs an Equal Volatility (EV) portfolio."""
        method_name = "EV"; print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years)...")
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        lookback_window = int(lookback_years * 12); min_lookback_required = 2
        if lookback_window < min_lookback_required: raise ValueError(f"Lookback window must be >= {min_lookback_required} months.")
        if lookback_window > len(self.returns_df): raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")
        weights_dict = {}; first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index): print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return
        first_calc_date = self.returns_df.index[first_calc_date_index]
        for date in self.returns_df.loc[first_calc_date:].index:
            current_loc = self.returns_df.index.get_loc(date); start_loc = current_loc - lookback_window + 1
            returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]; rolling_vol = returns_slice.std() * ANNUALIZATION_FACTOR_VOL
            rolling_vol[rolling_vol.abs() < SMALL_VOL_THRESHOLD] = SMALL_VOL_THRESHOLD; inv_vol = 1.0 / rolling_vol
            weights = inv_vol / inv_vol.sum(); weights = weights / weights.sum(); weights_dict[date] = weights
        if not weights_dict: print(f"Warning: No weights calculated for {method_name}."); return
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers)
        self._calculate_portfolio_returns(method_name, weights_df); print(f"{method_name} construction complete.")

    @staticmethod
    def _erc_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Objective: Minimize variance of percentage risk contributions."""
        portfolio_var = weights.T @ cov_matrix @ weights
        if portfolio_var < FLOAT_COMPARISON_THRESHOLD: return 0.0
        mrc = cov_matrix @ weights; percentage_rc = (weights * mrc) / portfolio_var; return np.var(percentage_rc)

    def _calculate_erc_weights(self, cov_matrix: pd.DataFrame, calculation_date: pd.Timestamp) -> pd.Series:
        """Internal: Calculates ERC weights via optimization."""
        if cov_matrix.shape != (self.num_assets, self.num_assets): raise ValueError(f"[{calculation_date.date()}] Cov matrix shape mismatch.")
        if not np.all(cov_matrix.columns == self.tickers) or not np.all(cov_matrix.index == self.tickers):
             cov_matrix = cov_matrix.reindex(index=self.tickers, columns=self.tickers)
             if cov_matrix.isnull().values.any(): raise ValueError(f"[{calculation_date.date()}] Cov matrix has NaNs after reindexing.")
        cov_matrix_values = cov_matrix.values
        try: np.linalg.cholesky(cov_matrix_values)
        except np.linalg.LinAlgError: cov_matrix_values += np.eye(self.num_assets) * COV_MATRIX_JITTER
        w0 = np.ones(self.num_assets) / self.num_assets; constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        result = minimize(self._erc_objective, w0, args=(cov_matrix_values,), method='SLSQP', bounds=bounds, constraints=constraints, tol=OPTIMIZER_TOLERANCE, options={'maxiter': 1000, 'ftol': OPTIMIZER_TOLERANCE})
        if not result.success: print(f"Warning [{calculation_date.date()}]: ERC optimization failed: {result.message}. Falling back to EW."); return pd.Series(w0, index=self.tickers)
        optimized_weights = result.x / result.x.sum(); return pd.Series(optimized_weights, index=self.tickers)

    def construct_erc(self, lookback_years: int = 2, use_ewm: bool = True) -> None:
        """Constructs an Equal Risk Contribution (ERC) portfolio."""
        method_name = f"ERC_{'EWM' if use_ewm else 'Simple'}"
        print(f"\nConstructing {method_name} portfolio (Lookback: {lookback_years} years)...")
        if lookback_years <= 0: raise ValueError("Lookback years must be positive.")
        lookback_window = int(lookback_years * 12); min_lookback_required = 2
        if lookback_window < min_lookback_required: raise ValueError(f"Lookback window must be >= {min_lookback_required} months.")
        if lookback_window > len(self.returns_df): raise ValueError(f"Lookback window ({lookback_window}) exceeds data length ({len(self.returns_df)}).")
        weights_dict = {}; first_calc_date_index = lookback_window - 1
        if first_calc_date_index >= len(self.returns_df.index): print(f"Warning: Not enough data for lookback {lookback_window}. Cannot construct {method_name}."); return
        first_calc_date = self.returns_df.index[first_calc_date_index]
        for date in self.returns_df.loc[first_calc_date:].index:
            current_loc = self.returns_df.index.get_loc(date); start_loc = current_loc - lookback_window + 1
            returns_slice = self.returns_df.iloc[start_loc : current_loc + 1]; cov_matrix = None; cov_calc_successful = False
            try:
                if use_ewm: ewm_cov = returns_slice.ewm(span=lookback_window, min_periods=lookback_window, adjust=True).cov(pairwise=True) * ANNUALIZATION_FACTOR_COV;
                else: cov_matrix = returns_slice.cov() * ANNUALIZATION_FACTOR_COV
                if use_ewm and date in ewm_cov.index.get_level_values(0): cov_matrix = ewm_cov.loc[date]
                if isinstance(cov_matrix, pd.DataFrame) and not cov_matrix.isnull().values.any() and cov_matrix.shape == (self.num_assets, self.num_assets): cov_calc_successful = True
                elif cov_matrix is not None: print(f"Warning [{date.date()}]: Invalid cov matrix ({'EWM' if use_ewm else 'Simple'}). Skipping.")
            except Exception as e: print(f"Error [{date.date()}] calculating cov matrix: {e}. Skipping.")
            if cov_calc_successful:
                try: erc_weights = self._calculate_erc_weights(cov_matrix, date); weights_dict[date] = erc_weights
                except Exception as e: print(f"Error [{date.date()}] calculating ERC weights: {e}. Skipping.")
        if not weights_dict: print(f"Warning: No weights calculated for {method_name}."); return
        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=self.tickers).fillna(0)
        self._calculate_portfolio_returns(method_name, weights_df); print(f"{method_name} construction complete.")

    def get_returns(self) -> pd.DataFrame:
        """Returns DataFrame of monthly returns for all constructed portfolios."""
        if not self.returns: print("Warning: No portfolio returns calculated."); return pd.DataFrame()
        return pd.DataFrame(self.returns)

    def get_weights(self, method_name: str) -> pd.DataFrame:
        """Returns DataFrame of portfolio weights for a specific method."""
        if method_name not in self.weights: available = list(self.weights.keys()); raise KeyError(f"Weights for '{method_name}' not found. Available: {available}")
        return self.weights[method_name]

    def get_total_return_indexes(self, base_value: int = 100) -> pd.DataFrame:
        """
        Calculates Total Return Indexes for constructed portfolios, starting
        from a common date (when *all* portfolios have data) and base value.
        """
        portfolio_returns_df = self.get_returns()
        if portfolio_returns_df.empty: print("Cannot generate TRI: No portfolio returns."); return pd.DataFrame()
        valid_returns = portfolio_returns_df.dropna(axis=1, how='all')
        if valid_returns.empty: print("Cannot generate TRI: All portfolios have NaN returns."); return pd.DataFrame()
        try:
            # Find first date where *all* remaining portfolios have non-NaN returns.
            # This inherently handles methods starting later due to lookbacks.
            common_start_date = valid_returns.dropna().index.min()
        except ValueError: print("Cannot generate TRI: No common start date found."); return pd.DataFrame()
        common_returns = valid_returns.loc[common_start_date:]
        if common_returns.empty: print(f"Cannot generate TRI: No common returns data from {common_start_date}."); return pd.DataFrame()
        tri = (1 + common_returns).cumprod() * base_value
        base_date = common_start_date - pd.Timedelta(days=1); base_row = pd.DataFrame({col: float(base_value) for col in tri.columns}, index=[base_date])
        final_tri = pd.concat([base_row, tri]).sort_index(); return final_tri.round(4)


class PortfolioAnalyzer:
    """
    Analyzes and compares performance metrics for portfolio strategies and
    underlying assets based on their monthly returns.
    """
    def __init__(self,
                 portfolio_returns: pd.DataFrame,
                 asset_returns: pd.DataFrame):
        """Initializes analyzer, combining returns & finding common date range."""
        if portfolio_returns.empty and asset_returns.empty: raise ValueError("Both portfolio and asset returns are empty.")
        self.all_returns = pd.concat([portfolio_returns, asset_returns], axis=1); self.all_returns.dropna(axis=1, how='all', inplace=True)
        if self.all_returns.empty: raise ValueError("Combined returns are empty after dropping NaN columns.")
        try:
            start_dt = self.all_returns.dropna(how='all').index.min(); end_dt = self.all_returns.dropna(how='all').index.max()
            self.all_returns = self.all_returns.loc[start_dt:end_dt]; self.common_start_date = self.all_returns.dropna().index.min()
            self.common_end_date = end_dt; self.all_returns = self.all_returns.loc[self.common_start_date:self.common_end_date]
            if self.all_returns.dropna().empty: raise ValueError("No overlapping non-NaN data.")
        except Exception as e: raise ValueError(f"Error determining common date range or slicing returns: {e}")
        print(f"\nInitializing Analyzer. Common period: {self.common_start_date.strftime('%Y-%m-%d')} to {self.common_end_date.strftime('%Y-%m-%d')}")
        self.metrics = ["Annualized Return", "Annualized Volatility", "Information Ratio", "Max Drawdown", "Calmar Ratio", "Max Drawdown Recovery (Months)", "% Positive Months", "Avg Positive Month Ret", "Avg Negative Month Ret"]

    def _calculate_annualized_return(self, returns_series: pd.Series) -> float:
        """Geometric annualized return."""
        if returns_series.empty or len(returns_series) == 0: return np.nan; num_years = len(returns_series) / 12.0
        if num_years <= 0: return np.nan;
        if ((1 + returns_series) <= 0).any(): return np.nan
        return ((1 + returns_series).prod())**(1 / num_years) - 1

    def _calculate_annualized_volatility(self, returns_series: pd.Series) -> float:
        """Annualized volatility."""
        if returns_series.empty or len(returns_series) < 2: return np.nan; return returns_series.std() * ANNUALIZATION_FACTOR_VOL

    def _calculate_information_ratio(self, ann_return: float, ann_vol: float) -> float:
        """Info Ratio (AnnRet / AnnVol)."""
        if ann_vol is None or abs(ann_vol) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(ann_vol): return np.nan
        return ann_return / ann_vol

    def _calculate_max_drawdown(self, returns_series: pd.Series) -> Tuple[float, Optional[float]]:
        """Max Drawdown (MDD) & recovery time in months."""
        if returns_series.empty or len(returns_series) < 1: return np.nan, np.nan
        tri = (1 + returns_series).cumprod(); base_idx = returns_series.index[0] - pd.Timedelta(days=1)
        tri.loc[base_idx] = 1.0; tri.sort_index(inplace=True); rolling_max = tri.cummax()
        drawdown = (tri / rolling_max) - 1.0; max_dd = drawdown.min()
        if max_dd >= -FLOAT_COMPARISON_THRESHOLD or pd.isnull(max_dd): return 0.0, 0.0
        trough_date = drawdown.idxmin(); peak_candidates = rolling_max.loc[:trough_date]; peak_date = peak_candidates[peak_candidates == peak_candidates.iloc[-1]].index[0]
        peak_value = tri.loc[peak_date]; duration_months: Optional[float] = np.nan
        try:
            recovery_candidates = tri.loc[trough_date:]; recovered_series = recovery_candidates[recovery_candidates >= peak_value - FLOAT_COMPARISON_THRESHOLD]
            if not recovered_series.empty:
                 recovery_date = recovered_series.index[0]; peak_loc = tri.index.get_loc(peak_date); recovery_loc = tri.index.get_loc(recovery_date)
                 duration_months = float(recovery_loc - peak_loc)
        except Exception as e: print(f"Warning: Error calculating recovery for series {returns_series.name or ''}: {e}"); duration_months = np.nan
        return max_dd, duration_months

    def _calculate_calmar_ratio(self, ann_return: float, max_drawdown: float) -> float:
        """Calmar Ratio (AnnRet / Abs(MaxDD))."""
        if max_drawdown is None or max_drawdown >= -FLOAT_COMPARISON_THRESHOLD or abs(max_drawdown) < FLOAT_COMPARISON_THRESHOLD or np.isnan(ann_return) or np.isnan(max_drawdown): return np.nan
        return ann_return / abs(max_drawdown)

    def _calculate_monthly_stats(self, returns_series: pd.Series) -> Tuple[float, float, float]:
        """% Positive Months, Avg Pos Ret, Avg Neg Ret."""
        if returns_series.empty: return np.nan, np.nan, np.nan; pos_mask = returns_series > 0; neg_mask = returns_series < 0
        pct_positive = pos_mask.mean() if len(returns_series) > 0 else np.nan; avg_positive = returns_series[pos_mask].mean() if pos_mask.any() else 0.0
        avg_negative = returns_series[neg_mask].mean() if neg_mask.any() else 0.0; return pct_positive, avg_positive, avg_negative

    def analyze_performance(self, periods_years: List[int] = [1, 3, 5, 10]) -> Dict[str, pd.DataFrame]:
        """Calculates performance metrics over specified periods and full sample."""
        period_results: Dict[str, pd.DataFrame] = {}; end_date = self.all_returns.index.max()
        periods_def = {'Full Sample': None};
        for p_yr in sorted(periods_years):
             if p_yr * 12 <= len(self.all_returns): periods_def[f'{p_yr}Y'] = p_yr * 12
        print(f"Analyzing periods: {list(periods_def.keys())}")
        for period_label, period_months in periods_def.items():
            print(f"  Analyzing Period: {period_label}"); period_data = {}
            for name in self.all_returns.columns:
                returns_slice = pd.Series(dtype=float); valid_slice = False
                if period_months:
                    start_date_period = end_date - pd.DateOffset(months=period_months - 1); actual_start_date = max(start_date_period, self.common_start_date)
                    _slice = self.all_returns.loc[actual_start_date:, name].dropna(); min_required_points = int(period_months * 0.9)
                    if len(_slice) >= min_required_points: returns_slice = _slice; valid_slice = True
                else: returns_slice = self.all_returns[name].dropna();
                if not returns_slice.empty: valid_slice = True
                if valid_slice:
                    ann_ret=self._calculate_annualized_return(returns_slice); ann_vol=self._calculate_annualized_volatility(returns_slice); info_ratio=self._calculate_information_ratio(ann_ret, ann_vol)
                    max_dd, recovery_months=self._calculate_max_drawdown(returns_slice); calmar=self._calculate_calmar_ratio(ann_ret, max_dd); pct_pos, avg_pos, avg_neg=self._calculate_monthly_stats(returns_slice)
                    metrics_values = {"Annualized Return": ann_ret, "Annualized Volatility": ann_vol, "Information Ratio": info_ratio, "Max Drawdown": max_dd, "Calmar Ratio": calmar, "Max Drawdown Recovery (Months)": recovery_months, "% Positive Months": pct_pos, "Avg Positive Month Ret": avg_pos, "Avg Negative Month Ret": avg_neg}
                else: metrics_values = {metric: np.nan for metric in self.metrics}
                period_data[name] = metrics_values
            period_summary_df = pd.DataFrame.from_dict(period_data, orient='index'); period_summary_df = period_summary_df.reindex(columns=self.metrics)
            formatted_df = period_summary_df.copy()
            format_pct = ["Annualized Return", "Annualized Volatility", "Max Drawdown", "% Positive Months", "Avg Positive Month Ret", "Avg Negative Month Ret"]
            format_ratio = ["Information Ratio", "Calmar Ratio"]; format_int = ["Max Drawdown Recovery (Months)"]
            for col in format_pct:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
            for col in format_ratio:
                if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
            for col in format_int:
                 if col in formatted_df.columns: formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x):,d}" if pd.notnull(x) and np.isfinite(x) else ("-" if pd.isnull(x) or not np.isfinite(x) else "Ongoing"))
            period_results[period_label] = formatted_df
        return period_results


def load_tri_data(csv_filepath: str,
                  date_column_name: str,
                  ticker_subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads Total Return Index data from a CSV file, sets DatetimeIndex,
    validates, converts to numeric, and optionally subsets tickers.
    """
    print(f"--- Loading Data from {csv_filepath} ---")
    try:
        tri_df = pd.read_csv(csv_filepath, index_col=date_column_name, parse_dates=[date_column_name], dayfirst=False)
        print(f"Loaded data shape: {tri_df.shape}")
        if not isinstance(tri_df.index, pd.DatetimeIndex): raise ValueError("Index is not DatetimeIndex.")
        if ticker_subset:
            print(f"Selecting subset: {', '.join(ticker_subset)}")
            missing = [t for t in ticker_subset if t not in tri_df.columns];
            if missing: warnings.warn(f"Tickers not found and ignored: {', '.join(missing)}")
            valid_subset = [t for t in ticker_subset if t in tri_df.columns];
            if not valid_subset: raise ValueError("None of the specified tickers found.")
            tri_df = tri_df[valid_subset]; print(f"Shape after subsetting: {tri_df.shape}")
            if tri_df.empty: raise ValueError("DataFrame empty after subsetting.")
        else: print("Using all available tickers.")
        tri_df.sort_index(inplace=True)
        for col in tri_df.columns: tri_df[col] = pd.to_numeric(tri_df[col], errors='coerce')
        all_nan_cols = tri_df.columns[tri_df.isnull().all()].tolist()
        if all_nan_cols: warnings.warn(f"Columns dropped due to all NaNs after numeric conversion: {', '.join(all_nan_cols)}"); tri_df.drop(columns=all_nan_cols, inplace=True)
        if tri_df.empty: raise ValueError("DataFrame empty after numeric conversion/dropping.")
        print("Data loaded and preprocessed successfully.")
        return tri_df
    except FileNotFoundError: print(f"ERROR: Data file not found at {csv_filepath}"); raise
    except KeyError: print(f"ERROR: Date column '{date_column_name}' not found."); raise
    except Exception as e: print(f"ERROR: Failed to load/process data: {e}"); raise


def generate_portfolio_report(portfolio_constructor: PortfolioConstructor,
                              analyzer_periods: List[int] = [1, 3, 5, 10],
                              output_filename: str = "portfolio_report.xlsx") -> None:
    """
    Generates a multi-sheet Excel report with portfolio analysis results.
    Includes TRI, Performance Summary, Asset Correlation, and Weights sheets.
    """
    print(f"\n--- Generating Final Report: {output_filename} ---")
    portfolio_tri_df = portfolio_constructor.get_total_return_indexes()
    portfolio_returns_df = portfolio_constructor.get_returns()
    asset_returns_df = portfolio_constructor.returns_df
    all_weights_dict = portfolio_constructor.weights
    if portfolio_tri_df.empty and not all_weights_dict and asset_returns_df.empty: print("No data available. Skipping report."); return

    # Perform Performance Analysis
    analyzer = None; summary_tables_dict: Dict[str, pd.DataFrame] = {}
    if not portfolio_returns_df.empty or not asset_returns_df.empty:
         try:
              common_idx = portfolio_returns_df.index.union(asset_returns_df.index).sort_values()
              port_ret_aligned = portfolio_returns_df.reindex(common_idx); asset_ret_aligned = asset_returns_df.reindex(common_idx)
              analyzer = PortfolioAnalyzer(port_ret_aligned, asset_ret_aligned)
              summary_tables_dict = analyzer.analyze_performance(periods_years=analyzer_periods)
         except ValueError as e: print(f"Could not initialize/run PortfolioAnalyzer: {e}")
         except Exception as e: print(f"Unexpected error during performance analysis: {e}")
    else: print("Skipping performance analysis: No returns.")

    # Calculate Asset Correlation
    asset_correlation_matrix = pd.DataFrame()
    if not asset_returns_df.empty:
        try: asset_correlation_matrix = asset_returns_df.corr().round(4); print("Calculated asset correlation matrix.")
        except Exception as e: print(f"Warning: Could not calculate asset correlation: {e}")

    # Write to Excel
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            sheet_num = 1
            # Sheet 1: TRI
            print(f"({sheet_num}) Writing Portfolio TRI sheet..."); sheet_name_tri = "Portfolio TRI"
            if not portfolio_tri_df.empty: portfolio_tri_df.to_excel(writer, sheet_name=sheet_name_tri, index=True); print(f"   - Done ({len(portfolio_tri_df)} rows)")
            else: print("   - Skipped (no data).")
            sheet_num += 1

            # Sheet 2: Performance Summary (Restored and Corrected)
            summary_sheet_name = "Performance Summary"
            print(f"({sheet_num}) Writing {summary_sheet_name} sheet...")
            if analyzer and summary_tables_dict:
                current_row = 0; period_order = ['Full Sample'] + [f'{p}Y' for p in sorted(analyzer_periods)]
                for period_label in period_order:
                    if period_label in summary_tables_dict:
                        df_to_write = summary_tables_dict[period_label]
                        if not df_to_write.empty:
                             # Write Title for the table section
                             title_df = pd.DataFrame([f"{period_label} Performance Summary"])
                             title_df.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=False, header=False)
                             current_row += 1
                             # Write the performance data table
                             df_to_write.to_excel(writer, sheet_name=summary_sheet_name, startrow=current_row, index=True, header=True)
                             # Update row pointer for the next table (+ header + data rows + 2 blank rows for spacing)
                             current_row += (len(df_to_write) + 3)
                             print(f"   - Added table for: {period_label}")
                        # else: # Optionally print skip message for empty tables
                             # print(f"   - Skipping empty table for: {period_label}")
                    # else: # Optionally print skip message if period results missing
                         # print(f"   - No results calculated for period: {period_label}")
                print(f"   - Done")
            else: print(f"   - Skipped (no analysis results).")
            sheet_num += 1

            # Sheet 3: Asset Correlation
            corr_sheet_name = "Asset Correlation"
            print(f"({sheet_num}) Writing {corr_sheet_name} sheet...")
            if not asset_correlation_matrix.empty: asset_correlation_matrix.to_excel(writer, sheet_name=corr_sheet_name, index=True, header=True); print(f"   - Done ({len(asset_correlation_matrix)} rows)")
            else: print(f"   - Skipped (no data/failed).")
            sheet_num +=1

            # Subsequent Sheets: Weights
            print(f"({sheet_num}+) Writing Weight sheets...")
            if not all_weights_dict: print("   - Skipped (no weights).")
            else:
                for method_name, weights_df in all_weights_dict.items():
                    sheet_name = f"{method_name} Weights"
                    if not weights_df.empty: weights_df.to_excel(writer, sheet_name=sheet_name, index=True); print(f"   - Writing sheet: {sheet_name} ({len(weights_df)} rows)")
                    else: print(f"   - Skipping {sheet_name} (no data).")
                    sheet_num +=1
        print(f"\nExcel report '{output_filename}' generated successfully.")
    except ImportError: print("\nError: 'openpyxl' required. Please install it (`pip install openpyxl`).");
    except Exception as e: print(f"\nAn error occurred while writing the Excel file: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main script to load data, run portfolio construction/analysis, and report.
    Requires a CSV file with Total Return Index data, where the first column
    is dates and subsequent columns are different assets/strategies.
    """

    # --- Configuration ---
    # *** USER: Set these parameters ***
    DATA_FILEPATH = 'your_data.csv' # <-- CHANGE THIS (e.g., 'data/strategy_tri.csv')
    DATE_COLUMN = 'Date'            # <-- CHANGE THIS (e.g., 'EffectiveDate')
    SELECTED_TICKERS: Optional[List[str]] = None # Example: ['US Equity', 'EM Equity', 'US Bonds'] or None for all
    ANALYSIS_PERIODS = [1, 3, 5, 10]        # Years for performance summary tables
    OUTPUT_FILENAME = "portfolio_analysis_report_final.xlsx" # Name of the output Excel file
    # ****************************

    try:
        # --- 1. Load Data ---
        tri_data = load_tri_data(
            csv_filepath=DATA_FILEPATH,
            date_column_name=DATE_COLUMN,
            ticker_subset=SELECTED_TICKERS
        )

        # --- 2. Portfolio Construction ---
        print("\n--- Running Portfolio Construction ---")
        constructor = PortfolioConstructor(tri_df=tri_data)
        constructor.construct_equal_weight()
        constructor.construct_equal_volatility(lookback_years=2)
        constructor.construct_erc(lookback_years=2, use_ewm=True)
        constructor.construct_erc(lookback_years=3, use_ewm=False)

        # --- 3. Generate Final Report ---
        generate_portfolio_report(
            constructor,
            analyzer_periods=ANALYSIS_PERIODS,
            output_filename=OUTPUT_FILENAME
        )

    except FileNotFoundError: print(f"\nFATAL ERROR: Input data file not found at '{DATA_FILEPATH}'.")
    except (ValueError, KeyError) as data_err: print(f"\nFATAL ERROR loading/processing data or initializing constructor: {data_err}")
    except ImportError as lib_err: print(f"\nFATAL ERROR: Missing library: {lib_err}. Please install 'openpyxl'.")
    except Exception as e: print(f"\nFATAL ERROR: An unexpected error occurred: {e}")