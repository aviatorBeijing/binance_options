import numpy as np,os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t
import matplotlib.pyplot as plt

def read_prices_from_csv(filename):
    data = pd.read_csv(filename)
    return data['close'].values

def negative_log_likelihood(params, log_returns):
    mu, sigma, nu = params
    n = len(log_returns)
    
    # Calculate z-scores
    z = (log_returns - mu) / sigma
    
    # Calculate negative log-likelihood
    log_likelihood = -np.sum(t.logpdf(z, df=nu) - np.log(sigma))
    return log_likelihood

def find_mle(log_returns):
    initial_guess = [np.mean(log_returns), np.std(log_returns), 5.0]
    bounds = [(-np.inf, np.inf), (1e-8, np.inf), (2.01, np.inf)]
    
    result = minimize(negative_log_likelihood, initial_guess, args=(log_returns,),
                      bounds=bounds, method='L-BFGS-B')
    
    mu_mle, sigma_mle, nu_mle = result.x
    return mu_mle, sigma_mle, nu_mle

def simulate_paths(S0, mu, sigma, nu, T, n_steps, n_paths):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0
    
    for i in range(1, n_steps + 1):
        z = t.rvs(df=nu, size=n_paths)  # Generate Student's t random variables
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

def calculate_annualized_volatility(paths, dt):
    log_returns = np.log(paths[1:] / paths[:-1])
    volatilities = np.std(log_returns, axis=1) / np.sqrt(dt)  # Annualize volatility
    return volatilities

def main():
    prices = read_prices_from_csv(os.getenv('USER_HOME','') + "/tmp/btc-usdt_1d.csv")
    log_returns = np.log(prices[1:] / prices[:-1])

    mu_mle, sigma_mle, nu_mle = find_mle(log_returns)

    print(f"Estimated mu: {mu_mle}")
    print(f"Estimated sigma: {sigma_mle}")
    print(f"Estimated nu (degrees of freedom): {nu_mle}")

    #####
    # Simulate 10 paths based on the MLE results
    S0 = prices[-1]  # Starting price (last available price)
    T = 1  # Time horizon (1 year)
    n_steps = 252  # Number of time steps (daily steps for 1 year)
    n_paths = 100  # Number of paths to simulate

    paths = simulate_paths(S0, mu_mle, sigma_mle, nu_mle, T, n_steps, n_paths)
    
    dt = T / n_steps
    volatilities = calculate_annualized_volatility(paths, dt)

    # Plot the simulated paths
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2,1,1)
    for i in range(n_paths):
        plt.plot(paths[:, i], lw=1.5)
    plt.title(f'Simulated GBM Paths with Student\'s t-distribution (n={n_paths})')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Price')
    
    plt.subplot(2,1,2)
    plt.plot(volatilities, lw=2, color='blue')

    fn = os.getenv('USER_HOME','') + '/tmp/gbm_fat_tail.png'
    plt.savefig(fn)
    print( '--saved:', fn)

if __name__ == "__main__":
    main()

