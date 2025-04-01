# Required libraries
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple

# --- Configuration ---
LOOKBACK_YEARS: float = 2.0
TRADING_DAYS_PER_YEAR: int = 252
# Use 'BME' for Business Month End frequency (updated from 'BM')
REBALANCE_FREQUENCY: str = 'BME'
METRICS_TO_ANALYZE: List[str] = [ # Defined with Z
    'Annualized Return',
    'Annualized Volatility',
    'Information Ratio',
    'Calmar Ratio'
]

# --- Data Handling ---
# Unchanged from previous version
class SubStrategyData:
    """
    Handles loading, validation, and basic processing of sub-strategy data.
    Assumes input data is a pandas DataFrame with a DatetimeIndex,
    columns representing strategy tickers, and values as Total Return Index (TRI).
    """
    def __init__(self, total_return_data: pd.DataFrame):
        if not isinstance(total_return_data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")
        if total_return_data.isnull().values.any():
            print("Warning: Input data contains NaNs. Consider filling or handling them appropriately.")
        self.total_returns: pd.DataFrame = total_return_data.sort_index()
        self.tickers: List[str] = self.total_returns.columns.tolist()
        self._calculate_daily_returns()

    def _calculate_daily_returns(self) -> None:
        self.daily_returns: pd.DataFrame = self.total_returns.pct_change().fillna(0)

    def get_daily_returns(self) -> pd.DataFrame:
        return self.daily_returns

    def get_total_returns(self) -> pd.DataFrame:
        return self.total_returns

    def get_tickers(self) -> List[str]:
        return self.tickers


# --- Core Logic: Portfolio Construction ---
# Unchanged from previous version
class PortfolioConstructor:
    """
    Constructs the portfolio based on specified weighting scheme and rebalancing.
    Implements equal volatility weighting, carefully avoiding lookahead bias.
    Runs backtest over the FULL period of the provided strategy_data.
    """
    def __init__(self, strategy_data: SubStrategyData, lookback_years: float, rebalance_freq: str):
        self.strategy_data: SubStrategyData = strategy_data
        self.lookback_window: int = int(lookback_years * TRADING_DAYS_PER_YEAR)
        self.min_lookback_periods: int = max(int(self.lookback_window * 0.5), 30)
        self.rebalance_freq: str = rebalance_freq
        self.daily_returns: pd.DataFrame = self.strategy_data.get_daily_returns()
        self.tickers: List[str] = self.strategy_data.get_tickers()
        self.portfolio_returns: Optional[pd.Series] = None
        self.weights_over_time: Optional[pd.DataFrame] = None

    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        if self.daily_returns.empty:
             return pd.DatetimeIndex([])
        potential_dates = self.daily_returns.resample(self.rebalance_freq).last().index
        min_date_for_rebalance = self.daily_returns.index[0] + pd.Timedelta(days=self.min_lookback_periods)
        valid_rebalance_dates = potential_dates[
            (potential_dates >= min_date_for_rebalance) &
            (potential_dates < self.daily_returns.index[-1])
        ]
        return valid_rebalance_dates

    def _calculate_volatility(self, data_subset: pd.DataFrame) -> pd.Series:
        if len(data_subset) < self.min_lookback_periods:
            return pd.Series(np.nan, index=data_subset.columns)
        rolling_std = data_subset.rolling(window=self.lookback_window, min_periods=self.min_lookback_periods).std()
        current_daily_std = rolling_std.iloc[-1]
        annualized_vol = current_daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        return annualized_vol

    def _calculate_equal_vol_weights(self, volatility: pd.Series) -> pd.Series:
        volatility = volatility.replace(0, np.inf).fillna(np.inf)
        inv_vol = 1.0 / volatility
        if (inv_vol <= 1e-9).all():
             weights = pd.Series(1.0 / len(volatility), index=volatility.index)
        else:
            weights = inv_vol / inv_vol.sum()
        return weights.fillna(0)

    def run_backtest(self) -> pd.Series:
        rebalance_dates = self._get_rebalance_dates()
        if rebalance_dates.empty:
             print("Warning: Not enough data or history to perform any rebalances based on the lookback period.")
             if self.daily_returns.empty:
                  print("Error: Cannot run fallback with empty daily returns data.")
                  self.portfolio_returns = pd.Series(dtype=float)
                  self.weights_over_time = pd.DataFrame()
                  return self.portfolio_returns
             initial_weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)
             aligned_returns, aligned_weights = self.daily_returns.align(initial_weights, axis=1, join='inner')
             self.portfolio_returns = aligned_returns.dot(aligned_weights)
             self.weights_over_time = pd.DataFrame([initial_weights.values], index=[self.daily_returns.index[0]], columns=initial_weights.index)
             return self.portfolio_returns

        all_portfolio_returns = []
        weights_dict = {}
        first_rebalance_date = rebalance_dates[0]
        initial_history = self.daily_returns.loc[:first_rebalance_date]
        initial_volatility = self._calculate_volatility(initial_history)
        current_weights = self._calculate_equal_vol_weights(initial_volatility)
        first_rebalance_idx_loc = self.daily_returns.index.searchsorted(first_rebalance_date)
        initial_period_end_date = self.daily_returns.index[first_rebalance_idx_loc]
        initial_period_returns = self.daily_returns.loc[:initial_period_end_date]
        aligned_returns, aligned_weights = initial_period_returns.align(current_weights, axis=1, join='inner')
        initial_portfolio_rets = aligned_returns.dot(aligned_weights)
        all_portfolio_returns.append(initial_portfolio_rets)
        weights_dict[self.daily_returns.index[0]] = current_weights

        for i, current_rebalance_date in enumerate(rebalance_dates):
            historical_data = self.daily_returns.loc[:current_rebalance_date]
            volatility = self._calculate_volatility(historical_data)
            current_weights = self._calculate_equal_vol_weights(volatility)
            weights_dict[current_rebalance_date] = current_weights
            period_start_index = self.daily_returns.index.searchsorted(current_rebalance_date) + 1
            if period_start_index >= len(self.daily_returns.index):
                continue
            period_start_date = self.daily_returns.index[period_start_index]
            if i + 1 < len(rebalance_dates):
                period_end_date = rebalance_dates[i+1]
            else:
                period_end_date = self.daily_returns.index[-1]
            if period_start_date > period_end_date:
                continue
            period_returns = self.daily_returns.loc[period_start_date:period_end_date]
            aligned_returns, aligned_weights = period_returns.align(current_weights, axis=1, join='inner')
            period_portfolio_returns = aligned_returns.dot(aligned_weights)
            all_portfolio_returns.append(period_portfolio_returns)

        if not all_portfolio_returns:
             raise ValueError("Backtest simulation resulted in no portfolio returns after processing rebalances. Check data range and parameters.")
        self.portfolio_returns = pd.concat(all_portfolio_returns).sort_index()
        self.weights_over_time = pd.DataFrame.from_dict(weights_dict, orient='index').sort_index()
        return self.portfolio_returns

    def get_weights_over_time(self) -> Optional[pd.DataFrame]:
        return self.weights_over_time


# --- Performance Calculation (MODIFIED) ---
class PerformanceCalculator:
    """
    Calculates various performance metrics from a daily return series.
    Can calculate metrics over a specified date range.
    """
    def __init__(self, daily_returns: pd.Series):
        """
        Initializes the PerformanceCalculator.

        Args:
            daily_returns: A pandas Series of daily returns for the FULL period.
        """
        # Store the full series, filtering will happen in calculation methods
        self.full_daily_returns = daily_returns.fillna(0)

    def _filter_returns(self,
                        start_date: Optional[Union[str, pd.Timestamp]] = None,
                        end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.Series:
        """ Filters the daily returns series based on optional start and end dates. """
        returns_to_process = self.full_daily_returns
        if start_date:
            # Ensure start_date is timestamp for comparison
            start_ts = pd.Timestamp(start_date)
            returns_to_process = returns_to_process.loc[returns_to_process.index >= start_ts]
        if end_date:
            # Ensure end_date is timestamp for comparison
            end_ts = pd.Timestamp(end_date)
            returns_to_process = returns_to_process.loc[returns_to_process.index <= end_ts]
        return returns_to_process

    def calculate_annualized_return(self,
                                    start_date: Optional[Union[str, pd.Timestamp]] = None,
                                    end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """ Calculates the geometric annualized return for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty:
            return 0.0

        cumulative_returns = (1 + daily_returns).cumprod()
        num_days = len(daily_returns)
        num_years = num_days / TRADING_DAYS_PER_YEAR if TRADING_DAYS_PER_YEAR > 0 else 0

        if num_years == 0 or cumulative_returns.empty:
            return 0.0
        total_return = cumulative_returns.iloc[-1] - 1
        base = 1 + total_return
        if base <= 0:
             if num_years > 0: return -1.0
             else: return 0.0
        if num_years <= 0: return 0.0
        return base ** (1 / num_years) - 1

    def calculate_annualized_volatility(self,
                                        start_date: Optional[Union[str, pd.Timestamp]] = None,
                                        end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """ Calculates the annualized volatility for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty or len(daily_returns) < 2:
            return 0.0
        return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_information_ratio(self,
                                    start_date: Optional[Union[str, pd.Timestamp]] = None,
                                    end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """ Calculates the Information Ratio for the specified period. """
        # Pass dates down to underlying calculations
        ann_ret = self.calculate_annualized_return(start_date, end_date)
        ann_vol = self.calculate_annualized_volatility(start_date, end_date)
        if ann_vol == 0 or np.isnan(ann_vol) or pd.isna(ann_vol):
            return np.nan
        return ann_ret / ann_vol

    def calculate_max_drawdown(self,
                               start_date: Optional[Union[str, pd.Timestamp]] = None,
                               end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """ Calculates the maximum drawdown for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty:
            return 0.0
        cumulative_returns = (1 + daily_returns).cumprod()
        if cumulative_returns.empty:
             return 0.0
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max).divide(rolling_max.replace(0, np.nan))
        if drawdown.empty or drawdown.isnull().all():
             return 0.0
        return drawdown.min(skipna=True)

    def calculate_calmar_ratio(self,
                               start_date: Optional[Union[str, pd.Timestamp]] = None,
                               end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """ Calculates the Calmar Ratio for the specified period. """
        # Pass dates down to underlying calculations
        ann_ret = self.calculate_annualized_return(start_date, end_date)
        max_dd = self.calculate_max_drawdown(start_date, end_date)
        if max_dd >= 0 or np.isnan(max_dd) or pd.isna(max_dd):
            return np.nan
        # Avoid division by zero if ann_ret is also zero (or max_dd is effectively zero)
        if abs(max_dd) < 1e-9:
             return np.nan
        return ann_ret / abs(max_dd)

    def calculate_metric(self,
                         metric_name: str,
                         start_date: Optional[Union[str, pd.Timestamp]] = None,
                         end_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """
        Calculates a specific metric by name for the specified period.

        Args:
            metric_name: The name of the metric to calculate.
            start_date: Optional start date for metric calculation period.
            end_date: Optional end date for metric calculation period.

        Returns:
            The calculated metric value (float).

        Raises:
            ValueError: If the metric_name is unknown.
        """
        # Pass dates to the specific calculation method
        if metric_name == 'Annualized Return':
            return self.calculate_annualized_return(start_date, end_date)
        elif metric_name == 'Annualized Volatility':
            return self.calculate_annualized_volatility(start_date, end_date)
        elif metric_name == 'Information Ratio':
            return self.calculate_information_ratio(start_date, end_date)
        elif metric_name == 'Calmar Ratio':
            return self.calculate_calmar_ratio(start_date, end_date)
        elif metric_name == 'Max Drawdown':
            return self.calculate_max_drawdown(start_date, end_date)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


# --- Shapley Value Attribution (MODIFIED) ---
class ShapleyAttribution:
    """
    Uses the SHAP library (KernelExplainer) to attribute portfolio metrics
    (calculated over a specific analysis period) to the underlying sub-strategies.
    Treats the full backtest + dated metric calculation process as the 'model'.
    """
    def __init__(self,
                 full_strategy_data: SubStrategyData,
                 portfolio_config: Dict, # Now expected to contain analysis dates if needed
                 metric_to_explain: str):
        """
        Initializes the ShapleyAttribution analyzer.

        Args:
            full_strategy_data: SubStrategyData object containing data for ALL strategies.
            portfolio_config: Dictionary containing configuration for PortfolioConstructor
                              AND optionally 'analysis_start_date', 'analysis_end_date'.
            metric_to_explain: The name of the metric (string) to calculate and explain.
        """
        self.full_strategy_data: SubStrategyData = full_strategy_data
        self.config: Dict = portfolio_config # Stores lookback, freq, and analysis dates
        self.metric_to_explain: str = metric_to_explain
        self.all_tickers: List[str] = self.full_strategy_data.get_tickers()
        self.num_strategies: int = len(self.all_tickers)
        # Extract analysis dates from config for use in prediction function
        self.analysis_start_date = self.config.get('analysis_start_date', None)
        self.analysis_end_date = self.config.get('analysis_end_date', None)


    def _prediction_function(self, X: np.ndarray) -> np.ndarray:
        """
        The core function required by SHAP's KernelExplainer.
        Runs the full portfolio backtest for a subset, then calculates the
        metric ONLY over the specified analysis date range.
        """
        results = []
        for i in range(X.shape[0]):
            coalition_vector = X[i, :]
            subset_indices = np.where(coalition_vector == 1)[0]
            if len(subset_indices) == 0:
                results.append(0.0)
                continue
            subset_tickers = [self.all_tickers[j] for j in subset_indices]

            try:
                if not subset_tickers:
                     metric_value = 0.0
                else:
                     subset_total_returns = self.full_strategy_data.get_total_returns()[subset_tickers]
                     if subset_total_returns.empty:
                          metric_value = 0.0
                     else:
                          subset_data = SubStrategyData(subset_total_returns)
                          temp_constructor = PortfolioConstructor(
                              strategy_data=subset_data,
                              lookback_years=self.config['lookback_years'],
                              rebalance_freq=self.config['rebalance_freq']
                          )
                          # Run backtest over the subset's full history
                          portfolio_returns_subset = temp_constructor.run_backtest()

                          if portfolio_returns_subset is None or portfolio_returns_subset.empty:
                              metric_value = 0.0
                          else:
                              # Calculate metric ONLY for the analysis period
                              calculator = PerformanceCalculator(portfolio_returns_subset)
                              metric_value = calculator.calculate_metric(
                                  self.metric_to_explain,
                                  start_date=self.analysis_start_date, # Use stored analysis dates
                                  end_date=self.analysis_end_date
                              )
                              if np.isnan(metric_value) or pd.isna(metric_value):
                                  metric_value = 0.0
            except Exception as e:
                if isinstance(e, ValueError) and "resulted in no portfolio returns" in str(e):
                     print(f"Info: Subset {subset_tickers} resulted in no returns during backtest. Assigning 0.0 for SHAP.")
                     metric_value = 0.0
                else:
                     print(f"Warning: Error calculating metric for subset {subset_tickers} during SHAP prediction: {type(e).__name__} - {e}. Assigning 0.0.")
                     metric_value = 0.0
            results.append(metric_value)
        return np.array(results)

    def calculate_shapley_values(self, num_samples: Union[int, str] = 'auto') -> tuple[np.ndarray, float]:
        """ Calculates Shapley values using SHAP KernelExplainer. """
        background_data = np.zeros((1, self.num_strategies))
        data_to_explain = np.ones((1, self.num_strategies))
        explainer = shap.KernelExplainer(self._prediction_function, background_data)
        shap_values_result = explainer.shap_values(data_to_explain, nsamples=num_samples)

        if shap_values_result is None:
             raise ValueError("SHAP KernelExplainer returned None for shap_values.")
        shap_values_out = shap_values_result[0] if isinstance(shap_values_result, list) and len(shap_values_result) > 0 else shap_values_result
        expected_value_out = explainer.expected_value[0] if isinstance(explainer.expected_value, (np.ndarray, list)) and len(explainer.expected_value) > 0 else explainer.expected_value

        if not isinstance(shap_values_out, np.ndarray) or shap_values_out.shape[-1] != self.num_strategies:
             print(f"Warning: Unexpected SHAP values structure: {type(shap_values_out)}. Attempting to handle.")
             if isinstance(shap_values_out, list) and len(shap_values_out) == self.num_strategies:
                 shap_values_out = np.array(shap_values_out)
             elif hasattr(shap_values_out, 'values') and isinstance(getattr(shap_values_out, 'values', None), np.ndarray): # Handle shap.Explanation object case
                 shap_values_out = shap_values_out.values[0] # Assuming single instance explained
                 if shap_values_out.shape[-1] != self.num_strategies:
                      raise ValueError(f"Could not obtain valid Shapley values array from Explanation object. Shape: {shap_values_out.shape}")
             else:
                 raise ValueError(f"Could not obtain valid Shapley values array. Shape: {getattr(shap_values_out, 'shape', 'N/A')}")
        if not isinstance(expected_value_out, (float, np.float_)):
             try: expected_value_out = float(expected_value_out)
             except (ValueError, TypeError): raise ValueError(f"Could not obtain valid float for expected_value. Got: {type(expected_value_out)}")
        return shap_values_out, expected_value_out


# --- Visualization Helper ---
# Unchanged from previous version
def visualize_shap_values(shap_values: np.ndarray,
                          expected_value: float,
                          features: List[str],
                          metric_name: str,
                          calculated_metric: float,
                          analysis_start: Optional[str]=None,
                          analysis_end: Optional[str]=None):
    """ Visualizes Shapley values using SHAP's plotting functions. """
    period_str = "full period"
    if analysis_start or analysis_end:
        start_str = str(analysis_start) if analysis_start else "start"
        end_str = str(analysis_end) if analysis_end else "end"
        period_str = f"period {start_str} to {end_str}"

    print(f"\n--- SHAP Attribution Analysis for: {metric_name} ({period_str}) ---")
    print(f"Base Value (Expected Value): {expected_value:.4f}")
    print(f"Actual Calculated Metric Value (for period): {calculated_metric:.4f}")
    print(f"Sum of Shapley Values + Base Value: {(expected_value + np.sum(shap_values)):.4f}")
    print("Individual Strategy Contributions (Shapley Values):")
    if shap_values.ndim > 1: shap_values_1d = shap_values.flatten()
    else: shap_values_1d = shap_values
    if len(features) != len(shap_values_1d):
         print(f"Warning: Mismatch between number of features ({len(features)}) and Shapley values ({len(shap_values_1d)}). Skipping detailed printout.")
    else:
         contribs = sorted(zip(features, shap_values_1d), key=lambda item: item[1], reverse=True)
         for feature, shap_val in contribs: print(f"  - {feature}: {shap_val:+.4f}")

    try:
        if shap_values.ndim > 1: shap_values_plot = shap_values[0]
        else: shap_values_plot = shap_values
        shap.force_plot(base_value=expected_value, shap_values=shap_values_plot, features=features, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for {metric_name} Contribution ({period_str})")
        plt.tight_layout(pad=1.5); plt.show()
    except Exception as e: print(f"Could not generate SHAP force plot for {metric_name}: {e}")


# --- Main Execution Block (MODIFIED) ---
if __name__ == "__main__":
    print("Starting Portfolio Analysis and SHAP Attribution...")

    # --- Define Analysis Period (Set to None for full period) ---
    # Example: Analyze metrics from 2021 onwards
    analysis_start_date: Optional[str] = "2021-01-01"
    analysis_end_date: Optional[str] = None # None means go to the end of data
    # Example: Full period analysis
    # analysis_start_date: Optional[str] = None
    # analysis_end_date: Optional[str] = None
    # --- End Define Analysis Period ---


    # 1. Load or Generate Sample Data
    print("Generating sample sub-strategy data...")
    num_years_data = 5
    end_ts = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_ts, periods=num_years_data * TRADING_DAYS_PER_YEAR, freq='B')
    tickers = ['Momentum', 'Value', 'Quality', 'LowVol', 'Growth']
    np.random.seed(42)
    daily_rets = pd.DataFrame(np.random.randn(len(dates), len(tickers)) * 0.01 + 0.0003, index=dates, columns=tickers)
    raw_data = (1 + daily_rets).cumprod() * 100
    print(f"Sample data generated with {len(tickers)} strategies from {dates[0].date()} to {dates[-1].date()}.")

    # 2. Initialize Data Handler
    strategy_data = SubStrategyData(raw_data)

    # 3. Configure and Run Backtest (Uses full data period)
    print(f"\nConfiguring portfolio: Lookback={LOOKBACK_YEARS}yrs, Rebalance='{REBALANCE_FREQUENCY}'")
    # Base config for portfolio construction
    portfolio_config = {
        'lookback_years': LOOKBACK_YEARS,
        'rebalance_freq': REBALANCE_FREQUENCY
    }
    portfolio_builder = PortfolioConstructor(
        strategy_data=strategy_data,
        lookback_years=portfolio_config['lookback_years'],
        rebalance_freq=portfolio_config['rebalance_freq']
    )
    print("Running portfolio backtest (over full data period)...")
    portfolio_daily_returns = portfolio_builder.run_backtest()
    print("Backtest complete.")

    # 4. Calculate Overall Portfolio Performance (for the specific analysis period)
    analysis_period_str = f"from {analysis_start_date or 'start'} to {analysis_end_date or 'end'}"
    print(f"\nCalculating overall portfolio performance metrics for analysis period ({analysis_period_str})...")
    # Use the full portfolio returns series, but calculate metrics for the analysis period
    overall_calculator = PerformanceCalculator(portfolio_daily_returns)
    metrics_results = {}
    for metric in METRICS_TO_ANALYZE:
        try:
            # Pass analysis dates to the calculator
            metrics_results[metric] = overall_calculator.calculate_metric(
                metric,
                start_date=analysis_start_date,
                end_date=analysis_end_date
            )
            print(f"  - {metric}: {metrics_results[metric]:.4f}")
        except ValueError as e:
            print(f"  - Could not calculate {metric}: {e}")

    # 5. Perform SHAP Attribution (based on metrics for the analysis period)
    print(f"\nStarting SHAP value calculation for metrics calculated over analysis period ({analysis_period_str})...")
    try: shap.initjs()
    except Exception as e: print(f"Note: shap.initjs() failed: {e}")

    num_shap_samples = 'auto'

    # Add analysis dates to config dict passed to ShapleyAttribution
    shap_config = portfolio_config.copy()
    shap_config['analysis_start_date'] = analysis_start_date
    shap_config['analysis_end_date'] = analysis_end_date

    for metric in METRICS_TO_ANALYZE:
        if metric not in metrics_results or pd.isna(metrics_results[metric]):
             print(f"\nSkipping SHAP analysis for {metric} due to invalid or missing calculated value for the analysis period.")
             continue

        print(f"\nCalculating SHAP values for: {metric} (using nsamples='{num_shap_samples}')...")
        # Pass the config WITH analysis dates
        shap_analyzer = ShapleyAttribution(
            full_strategy_data=strategy_data,
            portfolio_config=shap_config, # Use config with analysis dates
            metric_to_explain=metric
        )

        try:
             shap_values, expected_value = shap_analyzer.calculate_shapley_values(num_samples=num_shap_samples)
             # Visualize, passing analysis dates for context in the plot title/printout
             visualize_shap_values(
                 shap_values=shap_values,
                 expected_value=expected_value,
                 features=strategy_data.get_tickers(),
                 metric_name=metric,
                 calculated_metric=metrics_results[metric], # The metric calculated for the period
                 analysis_start=analysis_start_date,
                 analysis_end=analysis_end_date
             )
        except Exception as e:
             print(f"Error during SHAP calculation or visualization for {metric}: {type(e).__name__} - {e}")
             import traceback
             print(traceback.format_exc())

    print("\nAnalysis complete.")

