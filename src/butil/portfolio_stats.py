import pandas as pd
import numpy as np
import datetime

APPROX_DAILY_TRADING_HOURS = 24

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252
APPROX_MINUTES_PER_YEAR = APPROX_BDAYS_PER_YEAR * APPROX_DAILY_TRADING_HOURS * 12

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

MINUTELY = 'minutely'
DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    MINUTELY: APPROX_MINUTES_PER_YEAR,
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1
}


def _perf(returns: pd.Series, metric, rule='24/360'):
    # TODO: resample the returns to make sure its daily return
    try:
        ts = list(map(lambda e: datetime.datetime.strptime(
            e, '%Y-%m-%d %H:%M:%S').timestamp(), returns.index.values))
    except Exception as e:
        try:
            ts = list(map(lambda e: datetime.datetime.strptime(
                str(e)[:-10],'%Y-%m-%dT%H:%M:%S').timestamp(), returns.index.values))
        except Exception as _:
            pass

    returns = returns.resample('1D').agg("sum")
    
    unique, counts = np.unique(np.diff(ts), return_counts=True)
    nanmost = list(zip(unique, counts))
    nanmost = sorted(nanmost, key=lambda e: e[1], reverse=True)
    nanmost = nanmost[0][0]

    trading_hour_per_day = float( rule.split('/')[0] )
    trading_days_per_year = float( rule.split('/')[1])
    dts = ( trading_hour_per_day * 3600 )/nanmost
    #print('-- smallest time gap: ', dts, ' (days)')
    dts *= trading_days_per_year

    if metric == 'sortino':
        downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
        #downside = returns[returns<0].std(ddof=1)
        if downside == 0:
            #return np.nan # NaN causing JSON (front end) failutre: jsonify() failed on server side.
            return 0
        res = returns.mean() / downside
    elif metric == 'sharpe':
        divisor = returns.std(ddof=1)
        try:
            assert divisor > 0, "Impossible for std to be zero."
            res = returns.mean() / divisor
        except Exception as e:
            res = 0
    # Use 252 for daily only works if the "returns" are daily return!
    return res * np.sqrt(252)  # Convert to annual


def sharpe(returns: pd.Series, rule='24/360') -> float:
    return _perf(returns, 'sharpe', rule)


def sortino(returns: pd.Series, rule='24/360') -> float:
    return _perf(returns, 'sortino', rule)


def max_drawdowns(returns: pd.Series) -> float:
    out = np.empty(returns.shape[1:])
    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    np.nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()

    return out


def cum_returns(returns, starting_value=0, out=None):
    """
    Compute cumulative returns from simple returns.
    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::
            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
         - Also accepts two dimensional data. In this case, each column is
           cumulated.
    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.
    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out

def effective_years(returns, annualization=None):
    """
    Effective trading years (This is important when compare Cagr among different algo.)
    """
    if len(returns) < 1:
        return np.nan
    if annualization is None:  # By default, returns are assumed to be Daily
        returns = returns.dropna().resample('1D').sum()

    ann_factor = annualization_factor(DAILY, annualization)
    num_years = len(returns[returns != 0]) / ann_factor

    num_years = int(num_years*100)/100
    return num_years

def calc_cagr( returns, annualization=None): # Alias
    return annual_returns( returns, annualization=None)

def annual_returns(returns, annualization=None):
    """
        [Cagr]
        Determines the mean annual growth rate of returns. This is equivilent
        to the compound annual growth rate.
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Periodic returns of the strategy, noncumulative.
            - See full explanation in :func:`~empyrical.stats.cum_returns`.
        annualization : int, optional
            Suppress the `period` to convert
            returns into annual returns. Value should be the annual frequency of
            `returns`.
        Returns
        -------
        annual_return : float
            Annual Return as CAGR (Compounded Annual Growth Rate).
    """
    if len(returns) < 1:
        return np.nan
    if annualization is None:  # By default, returns are assumed to be Daily
        returns = returns.dropna().resample('1D').sum()

    ann_factor = annualization_factor(DAILY, annualization)
    #num_years = len(returns[returns != 0]) / ann_factor
    
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns)
    
    try:
        if num_years >0: # i.e., there are at least one buy/sell pair.
            r = (ending_value) ** (1 / num_years) - 1
        else:
            r = 1
    except Exception as e:
        print('-- ERROR: divided by zero')
        return 0
    return r

def simple_annual_returns(returns, annualization=None):
    """
        [total_return/num_years]
    """
    if len(returns) < 1:
        return np.nan
    if annualization is None:  # By default, returns are assumed to be Daily
        returns = returns.dropna().resample('1D').sum()

    ann_factor = annualization_factor(DAILY, annualization)
    #num_years = len(returns[returns != 0]) / ann_factor
    
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns)

    if num_years == 0: 
        print('*** num of years is zero!')
        return 0.0

    return (ending_value-1)/num_years

def annualization_factor(period, annualization) -> float:
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def cum_returns_final(returns) -> float:
    if len(returns) == 0:
        return np.nan
    result = np.nanprod(returns + 1, axis=0)
    return result


def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    def get_max_drawdown_underwater(underwater):
        """
        Determines peak, valley, and recovery dates given an 'underwater'
        DataFrame.
        An underwater DataFrame is a DataFrame that has precomputed
        rolling drawdown.
        Parameters
        ----------
        underwater : pd.Series
        Underwater returns (rolling drawdown) of a strategy.
        Returns
        -------
        peak : datetime
            The maximum drawdown's peak.
        valley : datetime
            The maximum drawdown's valley.
        recovery : datetime
            The maximum drawdown's recovery.
        """

        valley = underwater.idxmin()  # end of the period
        # Find first 0
        peak = underwater[:valley][underwater[:valley] == 0].index[-1]
        # Find last 0
        try:
            recovery = underwater[valley:][underwater[valley:] == 0].index[0]
        except IndexError:
            recovery = np.nan  # drawdown not recovered
        return peak, valley, recovery

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(returns) == 0)
                or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break
        
    drawdowns = sorted( drawdowns, key=lambda e: e[0], reverse=True)

    return drawdowns
