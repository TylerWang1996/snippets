# Required libraries
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple
import time # To estimate runtime
import traceback # For detailed error logging
# Required for Excel writing
from pandas import ExcelWriter # Make sure pandas is installed

# --- Configuration ---
# Portfolio Construction Parameters
PORTFOLIO_LOOKBACK_YEARS: float = 2.0 # Lookback for volatility calculation
REBALANCE_FREQUENCY: str = 'BME'     # Rebalancing frequency (Business Month End)

# Performance & SHAP Analysis Parameters
TRADING_DAYS_PER_YEAR: int = 252
METRICS_TO_ANALYZE: List[str] = [
    'Annualized Return',
    'Annualized Volatility',
    'Information Ratio',
    'Calmar Ratio'
]
# Define periods for the summary table generation {Label: Years_Lookback}
# None for Years_Lookback means the full sample period.
TABLE_ANALYSIS_PERIODS: Dict[str, Optional[int]] = {
    '1Y': 1,
    '3Y': 3,
    '5Y': 5,
    '10Y': 10,
    'Full': None
}
# SHAP Calculation Settings
NUM_SHAP_SAMPLES: Union[str, int] = 'auto' # Samples for KernelExplainer ('auto' or integer)

# Output File Configuration
OUTPUT_EXCEL_FILE = 'shapley_analysis_output.xlsx'
OUTPUT_SHEET_NAME = 'SHAP Analysis'
ROW_SPACING_BETWEEN_TABLES = 3 # Number of empty rows between tables in Excel


# --- Data Handling ---
class SubStrategyData:
    """
    Handles loading, validation, and processing of sub-strategy return data.

    Attributes:
        total_returns (pd.DataFrame): DataFrame of total return index values.
        daily_returns (pd.DataFrame): DataFrame of daily percentage returns.
        tickers (List[str]): List of strategy tickers (column names).
    """
    def __init__(self, total_return_data: pd.DataFrame):
        """
        Initializes the SubStrategyData object and calculates daily returns.

        Args:
            total_return_data (pd.DataFrame): DataFrame with a DatetimeIndex and
                                             Total Return Index values for columns (strategies).

        Raises:
            ValueError: If the input DataFrame index is not a DatetimeIndex.
        """
        if not isinstance(total_return_data.index, pd.DatetimeIndex):
            raise ValueError("Input data index must be a DatetimeIndex.")

        # Warn about NaNs only once when the main data is loaded, not during SHAP calls.
        if not any('_prediction_function' in frame.name for frame in traceback.extract_stack()):
             if total_return_data.isnull().values.any():
                  print("Warning: Input data contains NaNs. Ensure this is intended or handled.")

        self.total_returns: pd.DataFrame = total_return_data.sort_index()
        self.tickers: List[str] = self.total_returns.columns.tolist()
        self._calculate_daily_returns()


    def _calculate_daily_returns(self) -> None:
        """ Calculates daily percentage returns from the total return index. """
        self.daily_returns: pd.DataFrame = self.total_returns.pct_change().fillna(0)

    def get_daily_returns(self) -> pd.DataFrame:
        """ Returns the DataFrame of daily returns. """
        return self.daily_returns

    def get_total_returns(self) -> pd.DataFrame:
        """ Returns the original DataFrame of total return index values. """
        return self.total_returns

    def get_tickers(self) -> List[str]:
        """ Returns the list of strategy tickers. """
        return self.tickers

# --- Core Logic: Portfolio Construction ---
class PortfolioConstructor:
    """
    Constructs a portfolio using equal volatility weighting and periodic rebalancing.

    Runs the backtest over the full period of the provided strategy data. Avoids
    lookahead bias in volatility calculation.

    Attributes:
        portfolio_returns (Optional[pd.Series]): Daily returns of the constructed portfolio.
        weights_over_time (Optional[pd.DataFrame]): Weights assigned at each rebalance date.
    """
    def __init__(self,
                 strategy_data: SubStrategyData,
                 lookback_years: float,
                 rebalance_freq: str):
        """
        Initializes the PortfolioConstructor.

        Args:
            strategy_data (SubStrategyData): Object containing strategy return data.
            lookback_years (float): Lookback period (in years) for volatility calculation.
            rebalance_freq (str): Rebalancing frequency (pandas offset alias, e.g., 'BME').
        """
        self.strategy_data: SubStrategyData = strategy_data
        self.lookback_window_days: int = int(lookback_years * TRADING_DAYS_PER_YEAR)
        self.min_lookback_periods: int = max(int(self.lookback_window_days * 0.5), 30)
        self.rebalance_freq: str = rebalance_freq
        self.daily_returns: pd.DataFrame = self.strategy_data.get_daily_returns()
        self.tickers: List[str] = self.strategy_data.get_tickers()
        self.portfolio_returns: Optional[pd.Series] = None
        self.weights_over_time: Optional[pd.DataFrame] = None

    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """
        Determines the dates on which weights should be recalculated (rebalance dates).

        Returns:
            pd.DatetimeIndex: A sorted index of rebalancing dates.
        """
        if self.daily_returns.empty:
             return pd.DatetimeIndex([])

        potential_dates = self.daily_returns.resample(self.rebalance_freq).last().index

        if len(self.daily_returns.index) == 0:
             return pd.DatetimeIndex([])
        first_valid_data_date = self.daily_returns.index.min()
        min_date_for_rebalance = first_valid_data_date + pd.Timedelta(days=self.min_lookback_periods)

        last_data_date = self.daily_returns.index.max()
        valid_rebalance_dates = potential_dates[
            (potential_dates >= min_date_for_rebalance) &
            (potential_dates < last_data_date)
        ]
        return valid_rebalance_dates.sort_values()

    def _calculate_volatility(self, data_subset: pd.DataFrame) -> pd.Series:
        """
        Calculates annualized rolling volatility ending on the last date of data_subset.

        Args:
            data_subset (pd.DataFrame): DataFrame of daily returns for the lookback period.

        Returns:
            pd.Series: Annualized volatilities indexed by ticker.
        """
        if len(data_subset) < self.min_lookback_periods:
            return pd.Series(np.nan, index=data_subset.columns)
        rolling_std = data_subset.rolling(window=self.lookback_window_days,
                                          min_periods=self.min_lookback_periods).std()
        current_daily_std = rolling_std.iloc[-1]
        annualized_vol = current_daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        return annualized_vol

    def _calculate_equal_vol_weights(self, volatility: pd.Series) -> pd.Series:
        """
        Calculates normalized inverse volatility weights (Equal Volatility contribution).

        Args:
            volatility (pd.Series): Annualized volatilities indexed by ticker.

        Returns:
            pd.Series: Target portfolio weights for the next period.
        """
        processed_volatility = volatility.copy()
        processed_volatility[processed_volatility <= 1e-9] = np.inf
        processed_volatility = processed_volatility.fillna(np.inf)
        inv_vol = 1.0 / processed_volatility
        if (inv_vol.abs() < 1e-9).all():
             weights = pd.Series(1.0 / len(volatility), index=volatility.index)
        else:
            weights = inv_vol / inv_vol.sum()
        return weights.fillna(0)

    def run_backtest(self) -> pd.Series:
        """
        Runs the portfolio construction backtest over the full data period.

        Returns:
            pd.Series: Daily returns of the constructed equal-volatility portfolio.

        Raises:
            ValueError: If the simulation results in no returns after processing rebalances.
        """
        rebalance_dates = self._get_rebalance_dates()

        if rebalance_dates.empty:
             print("Warning: No valid rebalance dates found. Running with initial equal weights for the full period.")
             if self.daily_returns.empty:
                  print("Error: Cannot run fallback with empty daily returns data.")
                  self.portfolio_returns = pd.Series(dtype=float)
                  self.weights_over_time = pd.DataFrame()
                  return self.portfolio_returns
             initial_weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)
             aligned_returns, aligned_weights = self.daily_returns.align(initial_weights, axis=1, join='inner')
             self.portfolio_returns = aligned_returns.dot(aligned_weights)
             self.weights_over_time = pd.DataFrame([initial_weights.values],
                                                   index=[self.daily_returns.index.min()],
                                                   columns=initial_weights.index)
             return self.portfolio_returns

        all_portfolio_returns_list = []
        weights_dict = {}
        first_rebalance_date = rebalance_dates[0]
        initial_history_data = self.daily_returns.loc[:first_rebalance_date]
        initial_volatility = self._calculate_volatility(initial_history_data)
        initial_weights = self._calculate_equal_vol_weights(initial_volatility)
        weights_dict[self.daily_returns.index.min()] = initial_weights
        first_rebalance_idx_loc = self.daily_returns.index.searchsorted(first_rebalance_date)
        initial_period_end_date = self.daily_returns.index[first_rebalance_idx_loc]
        initial_period_returns_df = self.daily_returns.loc[:initial_period_end_date]
        aligned_returns, aligned_weights = initial_period_returns_df.align(initial_weights, axis=1, join='inner')
        initial_period_portfolio_returns = aligned_returns.dot(aligned_weights)
        all_portfolio_returns_list.append(initial_period_portfolio_returns)

        for i, current_rebalance_date in enumerate(rebalance_dates):
            historical_data = self.daily_returns.loc[:current_rebalance_date]
            current_volatility = self._calculate_volatility(historical_data)
            weights_for_next_period = self._calculate_equal_vol_weights(current_volatility)
            period_start_index = self.daily_returns.index.searchsorted(current_rebalance_date) + 1
            if period_start_index >= len(self.daily_returns.index): continue
            period_start_date = self.daily_returns.index[period_start_index]
            weights_dict[period_start_date] = weights_for_next_period
            next_rebalance_date = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns.index.max()
            period_end_date = next_rebalance_date
            if period_start_date > period_end_date: continue
            period_returns_df = self.daily_returns.loc[period_start_date:period_end_date]
            aligned_returns, aligned_weights = period_returns_df.align(weights_for_next_period, axis=1, join='inner')
            period_portfolio_returns = aligned_returns.dot(aligned_weights)
            all_portfolio_returns_list.append(period_portfolio_returns)

        if not all_portfolio_returns_list:
             print("Warning: Backtest simulation resulted in no portfolio returns after processing rebalances.")
             self.portfolio_returns = pd.Series(dtype=float) # Return empty series
             self.weights_over_time = pd.DataFrame()
             return self.portfolio_returns

        self.portfolio_returns = pd.concat(all_portfolio_returns_list).sort_index()
        self.weights_over_time = pd.DataFrame.from_dict(weights_dict, orient='index').sort_index()
        return self.portfolio_returns

    def get_weights_over_time(self) -> Optional[pd.DataFrame]:
        """ Returns the DataFrame of strategy weights at the start of each application period. """
        return self.weights_over_time

# --- Performance Calculation ---
class PerformanceCalculator:
    """
    Calculates standard performance metrics for a given return series.

    Allows calculation over a specified sub-period of the full return series.
    """
    def __init__(self, daily_returns_full_period: pd.Series):
        """
        Initializes the PerformanceCalculator.

        Args:
            daily_returns_full_period (pd.Series): Daily returns for the *entire* period.
        """
        self.full_daily_returns = daily_returns_full_period.fillna(0)

    def _filter_returns(self,
                        start_date: Optional[Union[str, pd.Timestamp]] = None,
                        end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.Series:
        """ Filters the stored daily returns series based on optional start/end dates. """
        returns_to_process = self.full_daily_returns
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None
        if start_ts:
            returns_to_process = returns_to_process.loc[returns_to_process.index >= start_ts]
        if end_ts:
            returns_to_process = returns_to_process.loc[returns_to_process.index <= end_ts]
        return returns_to_process

    def calculate_annualized_return(self, start_date=None, end_date=None) -> float:
        """ Calculates geometric annualized return for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty: return 0.0
        cumulative_returns = (1 + daily_returns).cumprod()
        num_days = len(daily_returns)
        num_years = num_days / TRADING_DAYS_PER_YEAR if TRADING_DAYS_PER_YEAR > 0 else 0
        if num_years <= 0 or cumulative_returns.empty: return 0.0
        total_return = cumulative_returns.iloc[-1] - 1
        base = 1 + total_return
        if base <= 0: return -1.0
        return base ** (1 / num_years) - 1

    def calculate_annualized_volatility(self, start_date=None, end_date=None) -> float:
        """ Calculates annualized volatility (std dev) for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty or len(daily_returns) < 2: return 0.0
        return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_information_ratio(self, start_date=None, end_date=None) -> float:
        """ Calculates Information Ratio (Ann. Return / Ann. Vol) for the period. """
        ann_ret = self.calculate_annualized_return(start_date, end_date)
        ann_vol = self.calculate_annualized_volatility(start_date, end_date)
        if pd.isna(ann_vol) or abs(ann_vol) < 1e-9: return np.nan
        return ann_ret / ann_vol

    def calculate_max_drawdown(self, start_date=None, end_date=None) -> float:
        """ Calculates the maximum peak-to-trough drawdown for the specified period. """
        daily_returns = self._filter_returns(start_date, end_date)
        if daily_returns.empty: return 0.0
        cumulative_returns = (1 + daily_returns).cumprod()
        if cumulative_returns.empty: return 0.0
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max).divide(rolling_max.replace(0, np.nan))
        if drawdown.empty or drawdown.isnull().all(): return 0.0
        return drawdown.min(skipna=True)

    def calculate_calmar_ratio(self, start_date=None, end_date=None) -> float:
        """ Calculates Calmar Ratio (Ann. Return / Abs(Max Drawdown)) for the period. """
        ann_ret = self.calculate_annualized_return(start_date, end_date)
        max_dd = self.calculate_max_drawdown(start_date, end_date)
        if pd.isna(max_dd) or max_dd >= 0 or abs(max_dd) < 1e-9: return np.nan
        return ann_ret / abs(max_dd)

    def calculate_metric(self, metric_name: str, start_date=None, end_date=None) -> float:
        """ Calculates a specific performance metric by name for the specified period. """
        if metric_name == 'Annualized Return': return self.calculate_annualized_return(start_date, end_date)
        elif metric_name == 'Annualized Volatility': return self.calculate_annualized_volatility(start_date, end_date)
        elif metric_name == 'Information Ratio': return self.calculate_information_ratio(start_date, end_date)
        elif metric_name == 'Calmar Ratio': return self.calculate_calmar_ratio(start_date, end_date)
        elif metric_name == 'Max Drawdown': return self.calculate_max_drawdown(start_date, end_date)
        else: raise ValueError(f"Unknown metric requested: {metric_name}")

# --- Shapley Value Attribution ---
class ShapleyAttribution:
    """
    Calculates Shapley values for sub-strategy contributions to a portfolio metric.

    Uses SHAP's KernelExplainer to estimate contributions based on simulating the
    portfolio construction process for various subsets of strategies. The final
    metric is calculated over a potentially specified analysis period.
    """
    def __init__(self,
                 full_strategy_data: SubStrategyData,
                 portfolio_config: Dict, # Includes lookback, freq, and analysis dates
                 metric_to_explain: str):
        """
        Initializes the ShapleyAttribution analyzer.

        Args:
            full_strategy_data (SubStrategyData): Data object for ALL strategies.
            portfolio_config (Dict): Configuration for portfolio construction, MUST include
                                     'lookback_years', 'rebalance_freq', and MAY include
                                     'analysis_start_date', 'analysis_end_date'.
            metric_to_explain (str): The name of the metric being attributed.
        """
        self.full_strategy_data: SubStrategyData = full_strategy_data
        self.config: Dict = portfolio_config
        self.metric_to_explain: str = metric_to_explain
        self.all_tickers: List[str] = self.full_strategy_data.get_tickers()
        self.num_strategies: int = len(self.all_tickers)
        self.analysis_start_date = self.config.get('analysis_start_date', None)
        self.analysis_end_date = self.config.get('analysis_end_date', None)
        if 'lookback_years' not in self.config or 'rebalance_freq' not in self.config:
             raise ValueError("Portfolio config dictionary must contain 'lookback_years' and 'rebalance_freq'.")

    def _prediction_function(self, X: np.ndarray) -> np.ndarray:
        """ SHAP's required prediction function. """
        output_metric_values = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            coalition_vector = X[i, :]
            subset_indices = np.where(coalition_vector == 1)[0]
            if len(subset_indices) == 0:
                output_metric_values[i] = 0.0
                continue
            subset_tickers = [self.all_tickers[j] for j in subset_indices]
            try:
                if not subset_tickers: metric_value = 0.0
                else:
                     subset_total_returns = self.full_strategy_data.get_total_returns()[subset_tickers]
                     if subset_total_returns.empty: metric_value = 0.0
                     else:
                          # Use a temporary SubStrategyData instance for the subset
                          subset_data = SubStrategyData(subset_total_returns)
                          temp_constructor = PortfolioConstructor(
                              strategy_data=subset_data,
                              lookback_years=self.config['lookback_years'],
                              rebalance_freq=self.config['rebalance_freq']
                          )
                          portfolio_returns_subset = temp_constructor.run_backtest()
                          if portfolio_returns_subset is None or portfolio_returns_subset.empty: metric_value = 0.0
                          else:
                              calculator = PerformanceCalculator(portfolio_returns_subset)
                              metric_value = calculator.calculate_metric(
                                  self.metric_to_explain,
                                  start_date=self.analysis_start_date,
                                  end_date=self.analysis_end_date
                              )
                              if pd.isna(metric_value): metric_value = 0.0
                output_metric_values[i] = metric_value
            except Exception as e:
                # Reduced verbosity for errors during SHAP sampling
                # print(f"\n--- ERROR during SHAP prediction for subset {subset_tickers} ---")
                # print(f"Metric: {self.metric_to_explain}")
                # print(f"Error Type: {type(e).__name__}, Message: {e}")
                # print("Assigning 0.0 for this subset and continuing...")
                # print("---")
                output_metric_values[i] = 0.0 # Assign 0 on error
        return output_metric_values

    def calculate_shapley_values(self, num_samples: Union[int, str] = 'auto') -> Tuple[np.ndarray, float]:
        """
        Calculates Shapley values using SHAP KernelExplainer.

        Args:
            num_samples (Union[int, str]): Number of coalitions to sample.

        Returns:
            Tuple[np.ndarray, float]: (1D array of SHAP values, expected_value).

        Raises:
            ValueError: If SHAP calculation fails or returns unexpected results.
        """
        background_data = np.zeros((1, self.num_strategies))
        data_to_explain = np.ones((1, self.num_strategies))
        try:
            explainer = shap.KernelExplainer(self._prediction_function, background_data)
            shap_values_result = explainer.shap_values(data_to_explain, nsamples=num_samples)
            base_value_result = explainer.expected_value
        except Exception as e:
            print(f"\n--- ERROR during SHAP KernelExplainer execution ---")
            print(f"Metric: {self.metric_to_explain}")
            print(f"Error Type: {type(e).__name__}, Message: {e}")
            print(traceback.format_exc())
            print("---")
            raise ValueError(f"SHAP KernelExplainer failed for metric {self.metric_to_explain}.") from e

        if shap_values_result is None: raise ValueError("SHAP KernelExplainer returned None for shap_values.")

        # --- Process SHAP outputs ---
        shap_values_processed = shap_values_result[0] if isinstance(shap_values_result, list) and len(shap_values_result) > 0 else shap_values_result
        expected_value_processed = base_value_result[0] if isinstance(base_value_result, (np.ndarray, list)) and len(base_value_result) > 0 else base_value_result

        # Ensure shap_values are numpy array
        if not isinstance(shap_values_processed, np.ndarray):
             if hasattr(shap_values_processed, 'values') and isinstance(getattr(shap_values_processed, 'values', None), np.ndarray):
                 shap_values_processed = shap_values_processed.values
             else:
                  try: shap_values_processed = np.array(shap_values_processed)
                  except Exception: pass

        # Ensure shap_values are 1D (N,)
        if isinstance(shap_values_processed, np.ndarray):
            if shap_values_processed.ndim == 2 and shap_values_processed.shape[0] == 1:
                shap_values_processed = shap_values_processed[0] # Squeeze (1, N) -> (N,)
            elif shap_values_processed.ndim == 0 and self.num_strategies == 1:
                 shap_values_processed = np.array([shap_values_processed.item()]) # Handle scalar

        # Final validation of shap_values shape
        if not isinstance(shap_values_processed, np.ndarray) or shap_values_processed.ndim != 1 or shap_values_processed.shape[0] != self.num_strategies:
             raise ValueError(f"Processed Shapley values have unexpected shape/type. "
                              f"Expected: ({self.num_strategies},), Got Shape: {getattr(shap_values_processed, 'shape', 'N/A')}, "
                              f"Type: {type(shap_values_processed)}")

        # Ensure expected_value is float
        if not isinstance(expected_value_processed, (float, np.number)):
             try: expected_value_processed = float(expected_value_processed)
             except (ValueError, TypeError): raise ValueError(f"Could not obtain valid float for expected_value. Got: {type(expected_value_processed)}")

        return shap_values_processed, expected_value_processed

# --- Visualization Helper ---
def visualize_shap_force_plot(shap_values: np.ndarray,
                              expected_value: float,
                              features: List[str],
                              metric_name: str,
                              calculated_metric: float,
                              analysis_start_date: Optional[pd.Timestamp]=None,
                              analysis_end_date: Optional[pd.Timestamp]=None):
    """ Generates and displays a SHAP force plot for a single analysis instance. """
    period_str = "full period"
    start_str = "start"
    end_str = "end"
    if analysis_start_date: start_str = analysis_start_date.date()
    if analysis_end_date: end_str = analysis_end_date.date()
    if analysis_start_date or analysis_end_date: period_str = f"period {start_str} to {end_str}"

    print(f"\n--- SHAP Attribution Visualization for: {metric_name} ({period_str}) ---")
    print(f"Base Value (Expected Value): {expected_value:.4f}")
    print(f"Actual Calculated Metric Value (for period): {calculated_metric:.4f}")
    print(f"Sum of Shapley Values + Base Value: {(expected_value + np.sum(shap_values)):.4f}")
    print("Individual Strategy Contributions (Shapley Values):")

    if shap_values.ndim != 1:
        print(f"Warning: visualize_shap_values expected 1D shap_values, got shape {shap_values.shape}. Attempting to flatten.")
        shap_values_1d = shap_values.flatten()
    else:
        shap_values_1d = shap_values

    if len(features) != len(shap_values_1d):
        print(f"Warning: Mismatch between features ({len(features)}) and Shapley values ({len(shap_values_1d)}). Cannot display detailed contributions.")
    else:
         contribs = sorted(zip(features, shap_values_1d), key=lambda item: item[1], reverse=True)
         for feature, shap_val in contribs: print(f"  - {feature}: {shap_val:+.4f}")

    try:
        # Ensure matplotlib is available and configured if running non-interactively
        shap.force_plot(base_value=expected_value, shap_values=shap_values_1d, features=features, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for {metric_name} Contribution ({period_str})")
        plt.tight_layout(pad=1.5);
        plt.show()

    except Exception as e:
        print(f"\n--- ERROR generating SHAP force plot for {metric_name} ({period_str}) ---")
        print(f"Error Type: {type(e).__name__}, Message: {e}")
        print("---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("="*80)
    print("Starting Portfolio Analysis and SHAP Attribution")
    print("="*80)
    overall_start_time = time.time()

    # --- Step 1: Define Analysis Period for Visualization ---
    VISUALIZATION_START_DATE: Optional[str] = "2021-01-01" # Set the desired start date 'YYYY-MM-DD' or None
    VISUALIZATION_END_DATE: Optional[str] = None       # Set the desired end date 'YYYY-MM-DD' or None

    # --- Examples for Setting Visualization Period ---
    # # Example: Visualize from Jan 1st, 2021 to the end of the data:
    # VISUALIZATION_START_DATE: Optional[str] = "2021-01-01"
    # VISUALIZATION_END_DATE: Optional[str] = None
    # # Example: Visualize the full data period:
    # VISUALIZATION_START_DATE: Optional[str] = None
    # VISUALIZATION_END_DATE: Optional[str] = None
    # # Example: Visualize the calendar year 2022:
    # VISUALIZATION_START_DATE: Optional[str] = "2022-01-01"
    # VISUALIZATION_END_DATE: Optional[str] = "2022-12-31"
    # --- End Examples ---


    # --- Step 2: Load or Generate Sample Data ---
    print("\n[Step 2] Loading/Generating Strategy Data...")
    num_years_data = 12
    data_end_date_ts = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    data_start_date_ts = data_end_date_ts - pd.DateOffset(years=num_years_data) + pd.Timedelta(days=1)
    dates_index = pd.date_range(start=data_start_date_ts, end=data_end_date_ts, freq='B')
    strategy_tickers = ['Momentum', 'Value', 'Quality', 'LowVol', 'Growth']
    np.random.seed(42)
    daily_returns_data = pd.DataFrame(np.random.randn(len(dates_index), len(strategy_tickers)) * 0.01 + 0.0003, index=dates_index, columns=strategy_tickers)
    total_return_index_data = (1 + daily_returns_data).cumprod() * 100
    strategy_data = SubStrategyData(total_return_index_data)
    data_start_date = strategy_data.get_daily_returns().index.min()
    data_end_date = strategy_data.get_daily_returns().index.max()
    print(f"  Data loaded spanning {data_start_date.date()} to {data_end_date.date()}.")

    # --- Step 3: Run Backtest (Full Period) ---
    print(f"\n[Step 3] Running Portfolio Backtest (Full Period)...")
    print(f"  Config: Vol Lookback={PORTFOLIO_LOOKBACK_YEARS}yrs, Rebalance='{REBALANCE_FREQUENCY}'")
    portfolio_construction_config = {'lookback_years': PORTFOLIO_LOOKBACK_YEARS, 'rebalance_freq': REBALANCE_FREQUENCY}
    portfolio_builder = PortfolioConstructor(strategy_data=strategy_data, **portfolio_construction_config)
    full_period_portfolio_returns = portfolio_builder.run_backtest()
    print("  Backtest complete.")

    # --- Step 4: Single Period Visualization Analysis ---
    print("\n[Step 4] Performing SHAP Analysis & Visualization for Specific Period...")
    visualization_start_ts = pd.Timestamp(VISUALIZATION_START_DATE) if VISUALIZATION_START_DATE else data_start_date
    visualization_end_ts = pd.Timestamp(VISUALIZATION_END_DATE) if VISUALIZATION_END_DATE else data_end_date
    visualization_start_ts = max(visualization_start_ts, data_start_date)
    visualization_end_ts = min(visualization_end_ts, data_end_date)

    if visualization_start_ts > visualization_end_ts:
         print(f"  ERROR: Visualization start date ({visualization_start_ts.date()}) is after end date ({visualization_end_ts.date()}). Skipping visualization.")
    else:
        viz_period_str = f"{visualization_start_ts.date()} to {visualization_end_ts.date()}"
        print(f"  Visualization Period: {viz_period_str}")
        try: shap.initjs()
        except Exception as e: print(f"  Note: shap.initjs() failed: {e}")

        viz_shap_config = portfolio_construction_config.copy()
        viz_shap_config['analysis_start_date'] = visualization_start_ts
        viz_shap_config['analysis_end_date'] = visualization_end_ts
        overall_perf_calculator = PerformanceCalculator(full_period_portfolio_returns)

        for metric in METRICS_TO_ANALYZE:
            print(f"\n  Analyzing Metric for Viz: {metric}...")
            try:
                actual_metric_viz = overall_perf_calculator.calculate_metric(
                     metric, start_date=visualization_start_ts, end_date=visualization_end_ts
                )
                if pd.isna(actual_metric_viz):
                     print(f"    Actual metric is NaN. Skipping SHAP visualization.")
                     continue

                shap_analyzer_viz = ShapleyAttribution(strategy_data, viz_shap_config, metric)
                shap_values_viz, expected_value_viz = shap_analyzer_viz.calculate_shapley_values(num_samples=NUM_SHAP_SAMPLES)

                visualize_shap_force_plot(
                    shap_values=shap_values_viz, expected_value=expected_value_viz,
                    features=strategy_data.get_tickers(), metric_name=metric,
                    calculated_metric=actual_metric_viz,
                    analysis_start_date=visualization_start_ts, analysis_end_date=visualization_end_ts
                )
            except Exception as e:
                 print(f"    ERROR during SHAP calculation or visualization for {metric} (Viz Period): {type(e).__name__} - {e}")

    # --- Step 5: Multi-Period Table Calculation ---
    print("\n[Step 5] Calculating SHAP Values for Multi-Period Summary Table...")
    print(f"  Table Periods: {', '.join(TABLE_ANALYSIS_PERIODS.keys())}")
    print(f"  SHAP Samples: {NUM_SHAP_SAMPLES}")
    print("  (This runs separately from the visualization step)")

    shap_results_table_data = {metric: {} for metric in METRICS_TO_ANALYZE}
    actual_metrics_table_data = {metric: {} for metric in METRICS_TO_ANALYZE}

    for period_label, years_lookback in TABLE_ANALYSIS_PERIODS.items():
        period_start_time = time.time()
        print(f"\n-- Processing Table Period: {period_label} --")
        current_period_end_ts = data_end_date
        if years_lookback is not None:
            current_period_start_ts = current_period_end_ts - pd.DateOffset(years=years_lookback) + pd.Timedelta(days=1)
            current_period_start_ts = max(current_period_start_ts, data_start_date)
        else:
            current_period_start_ts = data_start_date
        # current_period_str = f"{current_period_start_ts.date()} to {current_period_end_ts.date()}" # Keep console cleaner
        # print(f"  Date Range: {current_period_str}") # Keep console cleaner

        current_table_shap_config = portfolio_construction_config.copy()
        current_table_shap_config['analysis_start_date'] = current_period_start_ts
        current_table_shap_config['analysis_end_date'] = current_period_end_ts

        for metric in METRICS_TO_ANALYZE:
            metric_start_time = time.time()
            print(f"   * Calculating Table SHAP for: {metric}...")
            try:
                 actual_metric_value = overall_perf_calculator.calculate_metric(
                     metric, start_date=current_period_start_ts, end_date=current_period_end_ts
                 )
                 actual_metrics_table_data[metric][period_label] = actual_metric_value
                 if pd.isna(actual_metric_value):
                      print(f"     Actual metric is NaN. Storing NaNs for SHAP.")
                      shap_results_table_data[metric][period_label] = {ticker: np.nan for ticker in strategy_tickers}
                      continue

                 shap_analyzer_table = ShapleyAttribution(strategy_data, current_table_shap_config, metric)
                 shap_values, _ = shap_analyzer_table.calculate_shapley_values(num_samples=NUM_SHAP_SAMPLES)
                 shap_results_table_data[metric][period_label] = dict(zip(strategy_data.get_tickers(), shap_values))
                 # print(f"     Calculation complete ({time.time() - metric_start_time:.2f}s)") # Keep console cleaner
            except Exception as e:
                 print(f"     ERROR during Table SHAP calculation for {metric} ({period_label}): {type(e).__name__} - {e}. Storing NaNs.")
                 shap_results_table_data[metric][period_label] = {ticker: np.nan for ticker in strategy_tickers}

        print(f"-- Period {period_label} table calculations complete ({time.time() - period_start_time:.2f}s) --")

    # --- Step 6: Format and Write Results to Excel ---
    # Note: Requires 'openpyxl' engine: pip install openpyxl
    print(f"\n[Step 6] Writing Summary Tables to Excel file: {OUTPUT_EXCEL_FILE}...")
    try:
        with ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
            current_row = 0 # Keep track of where to write next table

            # --- Write Actual Metrics Table ---
            print("  Writing Actual Metrics table...")
            actual_metrics_df = pd.DataFrame(actual_metrics_table_data)
            if not actual_metrics_df.empty:
                table_period_order_actuals = list(TABLE_ANALYSIS_PERIODS.keys())
                actual_metrics_df = actual_metrics_df.reindex(index=table_period_order_actuals)

                # Write DataFrame FIRST to create the sheet
                actual_metrics_df.to_excel(
                    writer,
                    sheet_name=OUTPUT_SHEET_NAME,
                    startrow=current_row + 1, # Leave row 0 for the title
                    startcol=0,
                    index=True,
                    header=True
                )
                # Get the sheet object AFTER it's created
                sheet = writer.sheets[OUTPUT_SHEET_NAME]
                # Write title in the first row
                sheet.cell(row=current_row + 1, column=1, value="Actual Portfolio Metric Values") # +1 because excel is 1-based index for cells

                # Update row counter for next table
                current_row += len(actual_metrics_df) + ROW_SPACING_BETWEEN_TABLES + 2 # +2 for title and header row of the table just written
            else:
                 print("  No actual metric data available to write.")
                 # If no actual metrics, we might need to handle sheet creation differently if SHAP data also fails
                 # For now, assume SHAP data might exist


            # --- Write SHAP Tables (One per Metric) ---
            print("  Restructuring and writing SHAP tables...")
            shap_table_data_restructured = {}
            for metric, period_data in shap_results_table_data.items():
                for period_label, ticker_values in period_data.items():
                    if isinstance(ticker_values, dict):
                        for ticker, shap_value in ticker_values.items():
                            if ticker not in shap_table_data_restructured: shap_table_data_restructured[ticker] = {}
                            shap_table_data_restructured[ticker][(metric, period_label)] = shap_value
                    else:
                         for ticker in strategy_tickers:
                             if ticker not in shap_table_data_restructured: shap_table_data_restructured[ticker] = {}
                             shap_table_data_restructured[ticker][(metric, period_label)] = np.nan

            shap_results_df_full = pd.DataFrame.from_dict(shap_table_data_restructured, orient='index')

            if not shap_results_df_full.empty:
                shap_results_df_full.columns = pd.MultiIndex.from_tuples(
                    shap_results_df_full.columns, names=['Metric', 'Period']
                )
                table_period_order = list(TABLE_ANALYSIS_PERIODS.keys())

                # Loop through each metric to create separate tables
                for metric in METRICS_TO_ANALYZE:
                    if metric in shap_results_df_full.columns.get_level_values('Metric'):
                        print(f"    Writing SHAP table for: {metric}...")
                        # Select columns for the current metric
                        metric_shap_df = shap_results_df_full[metric]
                        # Reorder columns based on defined period order
                        metric_shap_df = metric_shap_df.reindex(columns=table_period_order)

                        # Write DataFrame for this metric first (if sheet doesn't exist yet)
                        # Or write title first if sheet already exists
                        if OUTPUT_SHEET_NAME not in writer.sheets:
                             # This case should ideally not happen if actual_metrics_df was written
                             print(f"      Sheet '{OUTPUT_SHEET_NAME}' not found, writing DF first for {metric}")
                             metric_shap_df.to_excel(writer, sheet_name=OUTPUT_SHEET_NAME, startrow=current_row + 1, startcol=0)
                             sheet = writer.sheets[OUTPUT_SHEET_NAME]
                             sheet.cell(row=current_row + 1, column=1, value=f"SHAP Value Contribution: {metric}")
                        else:
                             # Sheet exists, write title first
                             sheet = writer.sheets[OUTPUT_SHEET_NAME]
                             sheet.cell(row=current_row + 1, column=1, value=f"SHAP Value Contribution: {metric}")
                             metric_shap_df.to_excel(writer, sheet_name=OUTPUT_SHEET_NAME, startrow=current_row + 1, startcol=0)

                        # Update row counter
                        current_row += len(metric_shap_df) + ROW_SPACING_BETWEEN_TABLES + 2 # +2 for title and header
                    else:
                        print(f"    No SHAP data found for metric: {metric}")
            else:
                print("  No SHAP data available to write.")

        print(f"\n  Successfully wrote results to {OUTPUT_EXCEL_FILE}")

    except ImportError:
        print("\n--- ERROR writing to Excel ---")
        print("Could not import 'openpyxl'. Please install it to write Excel files:")
        print("  pip install openpyxl")
        print("---")
    except Exception as e:
        print(f"\n--- ERROR writing to Excel file {OUTPUT_EXCEL_FILE} ---")
        print(f"Error Type: {type(e).__name__}, Message: {e}")
        print(traceback.format_exc())
        print("---")


    # --- Analysis Completion ---
    total_runtime = time.time() - overall_start_time
    print("\n" + "="*80)
    print(f"Analysis complete. Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes).")
    print("="*80)

