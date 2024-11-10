import numpy as np,os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt

from strategy.gamma_scalping._configs import *

plt.style.use('fivethirtyeight')

def read_prices_from_csv(sym):
    from spot_trading.market_data import get_filebased_kline
    data, _ts = get_filebased_kline( sym )
    if 'timestamp' not in data:
        data = data.reset_index()
    dates = pd.date_range( end=data['timestamp'].iloc[-1],periods=data.shape[0] )
    return data['close'].values, dates 

# Negative log-likelihood function for MLE using Merton Jump Diffusion Model
def negative_log_likelihood(params, log_returns, dt):
    mu, sigma, lambda_, mu_j, sigma_j = params
    
    # Compute the log-likelihood
    jumps = poisson.pmf(0, lambda_ * dt) * norm.logpdf(log_returns, loc=mu * dt, scale=sigma * np.sqrt(dt))
    for k in range(1, 10):  # Consider up to 10 jumps in one step
        log_jumps = norm.logpdf(log_returns - k * mu_j, loc=(mu - k * lambda_ * mu_j) * dt, scale=np.sqrt(sigma**2 * dt + k * sigma_j**2))
        jumps += poisson.pmf(k, lambda_ * dt) * log_jumps
    
    # FIXME: nan values found in "jumps"
    #for v in jumps:
    #    if np.isnan(v): print( v )

    nll = -np.sum(np.log(jumps))
    return nll

# Function to find the MLE estimates for Merton Jump Diffusion Model
def find_mle(log_returns, dt):
    initial_guess = [np.mean(log_returns) / dt, np.std(log_returns) / np.sqrt(dt), 0.1, 0, 0.01]    # Initial guesses 
    bounds = [(-np.inf, np.inf), (1e-8, np.inf), (1e-8, np.inf), (-np.inf, np.inf), (1e-8, np.inf)]
    
    # Minimize the negative log-likelihood
    result = minimize(  negative_log_likelihood, 
                        initial_guess, 
                        args=(log_returns, dt),
                        bounds=bounds, 
                        method='L-BFGS-B')
    
    mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle = result.x

    if False:
        print('-- MLE results:')
        print(f"\tEstimated mu: {mu_mle}")
        print(f"\tEstimated sigma: {sigma_mle}")
        print(f"\tEstimated lambda (jump intensity): {lambda_mle}")
        print(f"\tEstimated mu_j (mean jump size): {mu_j_mle}")
        print(f"\tEstimated sigma_j (jump volatility): {sigma_j_mle}")

    return mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle

# Merton Jump Diffusion Model
def simulate_jump_diffusion_paths(S0, mu, sigma, lambda_, mu_j, sigma_j, T, n_steps, n_paths):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    
    for i in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)  # Brownian motion component
        jumps = np.random.poisson(lambda_ * dt, n_paths)  # Number of jumps
        jump_sizes = np.random.normal(loc=mu_j, scale=sigma_j, size=n_paths) * jumps  # Jump magnitude
        
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z + jump_sizes)
    
    return paths

def calculate_annualized_volatility(paths, tS):
    log_returns = np.log(paths[1:] / paths[:-1])
    volatilities = np.std(log_returns, axis=1) * np.sqrt(tS) 
    return volatilities

def calc_log_returns(prices):
    return np.log(prices[1:] / prices[:-1])

def calibrate_and_generate(prices, n_paths=100, horizon=1, t0:str=''):
    T = horizon  # Time horizon (1 year)

    # Calibration
    dt = 1/nDays  # Time step size
    lrtns = calc_log_returns(prices)
    mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle = find_mle( lrtns, dt)

    S0 = prices[-1]  # Starting price (last available price)
    n_steps = nDays * T

    paths = simulate_jump_diffusion_paths(S0, mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle, T, n_steps, n_paths)

    dates = list( range( n_steps ) )
    if t0:
        dates = pd.date_range(start=t0, periods=n_steps+1 )
    return ( paths, sigma_mle, dates )

def bootstrap_history(prices, dates, tailn=10):
    sigmas = []
    for i in range(1, tailn+1):
        xdates = dates[:-(tailn+1-i)]
        xprices = prices[:-(tailn+1-i)]
        mle_sigma, _, mean_sigma, std_sigma, p68_sigma,_,_ = _sim(
            xprices, xdates, nDays, 
            n_paths=100 )
        sigmas += [{
            'calc_date': xdates[-1],
            'mle_sigma': mle_sigma,
            'sim_mean': mean_sigma,
            'sim_std':  std_sigma }]
    df = pd.DataFrame.from_records(sigmas)
    df.calc_date = df.calc_date.apply(pd.Timestamp)
    df.set_index('calc_date',inplace=True)

    from tabulate import tabulate
    print( tabulate(df,headers="keys"))
    fn = os.getenv("USER_HOME","")
    fn = f'{fn}/tmp/merton_jump_bootstraping.png'
    
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(18, 20))
    
    df[['mle_sigma']].plot(ax=ax1,label='sigma from MLE calibration')
    df[['sim_mean']].plot(ax=ax1,label='avg. sigma from montecarlo')
    (df.sim_mean+df.sim_std).plot(ax=ax1,linestyle='--',label='+1s')
    (df.sim_mean-df.sim_std).plot(ax=ax1,linestyle='--',label='-1s')
    ax1.set_title('Bootstrapping Historical MLE Sigma')
    ax1.legend()

    ax2.plot(dates[:-tailn],prices[:-tailn],label='history')
    ax2.plot(dates[-tailn:],prices[-tailn:],label="bootstrapping")
    ax2.set_title('Market Closes')
    ax2.legend()

    plt.savefig(fn)
    print('-- saved:',fn)
    
    
def _sim(prices,dates,tScale,n_paths = 100):
    np.random.seed(371723)
    paths, sigma_mle, gen_dates = calibrate_and_generate(prices, n_paths=n_paths, t0=str(dates[-1]) )
    volatilities = calculate_annualized_volatility(paths, tScale)

    mean_volatility = np.mean(volatilities)
    std_volatility = np.std(volatilities)
    p68 = np.percentile(volatilities,68)
    return sigma_mle,volatilities,\
            mean_volatility,std_volatility,p68,\
            paths,gen_dates

import click
@click.command()
@click.option('--sym', default='btc')
def main(sym):
    x = 5
    prices, dates = read_prices_from_csv(sym)
    print(dates[0],'~',dates[-1])
    tScale = nDays    # Indicates the time scale in the data "prices"
    prices = prices[-nDays*x:]
    dates  = dates[-nDays*x:]
    n_paths = 100

    mle_sigma,\
        volatilities, mean_volatility,std_volatility,p68, \
            paths,gen_dates = _sim(
        prices,dates,tScale,n_paths = n_paths)
    print('-- Simulated sigma (mean,std,p68):', 
            f'{mean_volatility:.2f}', 
            f'{std_volatility:.2f}', 
            f'{p68:.2f}')

    plt.figure(figsize=(18, 20))
    
    plt.subplot(3,1,1)
    plt.plot(dates, prices)
    plt.title(f'{sym.upper()} Market Prices ({x}-Yr)')

    plt.subplot(3,1,2)
    for i in range(n_paths):
        plt.plot(gen_dates, paths[:, i], lw=1.5)
    plt.title(f'Merton Jump Diffusion Model (n={n_paths})')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Price')

    plt.subplot(3,1,3)
    plt.plot(volatilities, lw=2, color='blue')
    plt.axhline(mean_volatility + std_volatility, color='red', linestyle='--', label='+1σ Bound')
    plt.axhline(mean_volatility - std_volatility, color='green', linestyle='--', label='-1σ Bound')
    plt.axhline(mean_volatility, color='gray', linestyle='-', label='Mean Volatility')
    plt.title('Estimated Annualized Vol.')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Annualized Vol.')
    
    fn = os.getenv('USER_HOME','') + f'/tmp/merton_jump_model_{sym}.png'
    plt.savefig(fn)
    print( '-- saved:', fn)

    # Bootstrapping
    bootstrap_history(prices,dates,tailn=10) #365)

if __name__ == "__main__":
    print('***** Volatility Estimation by MLE (fitting Merton-Jump model) *****')
    main()

