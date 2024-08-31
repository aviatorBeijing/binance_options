import numpy as np,os
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def read_prices_from_csv(filename):
    data = pd.read_csv(filename)
    return data['close'].values

# Negative log-likelihood function for MLE using Merton Jump Diffusion Model
def negative_log_likelihood(params, log_returns, dt):
    mu, sigma, lambda_, mu_j, sigma_j = params
    
    # Compute the log-likelihood
    jumps = poisson.pmf(0, lambda_ * dt) * norm.logpdf(log_returns, loc=mu * dt, scale=sigma * np.sqrt(dt))
    for k in range(1, 10):  # Consider up to 10 jumps in one step
        log_jumps = norm.logpdf(log_returns - k * mu_j, loc=(mu - k * lambda_ * mu_j) * dt, scale=np.sqrt(sigma**2 * dt + k * sigma_j**2))
        jumps += poisson.pmf(k, lambda_ * dt) * log_jumps
    nll = -np.sum(np.log(jumps))
    return nll

# Function to find the MLE estimates for Merton Jump Diffusion Model
def find_mle(log_returns, dt):
    # Initial guesses 
    initial_guess = [np.mean(log_returns) / dt, np.std(log_returns) / np.sqrt(dt), 0.1, 0, 0.01]
    
    bounds = [(-np.inf, np.inf), (1e-8, np.inf), (1e-8, np.inf), (-np.inf, np.inf), (1e-8, np.inf)]
    
    # Minimize the negative log-likelihood
    result = minimize(negative_log_likelihood, initial_guess, args=(log_returns, dt),
                      bounds=bounds, method='L-BFGS-B')
    
    mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle = result.x
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

def calculate_annualized_volatility(paths, dt):
    log_returns = np.log(paths[1:] / paths[:-1])
    volatilities = np.std(log_returns, axis=1) / np.sqrt(dt) 
    return volatilities

def main():
    prices = read_prices_from_csv(os.getenv('USER_HOME','') + "/tmp/btc-usdt_1d.csv")[-365*5:]
    log_returns = np.log(prices[1:] / prices[:-1])
    
    nDays = 365 # Trading days per year

    # Time step size
    dt = 1/nDays  # indicates "daily"
    
    # Calibration
    mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle = find_mle(log_returns, dt)

    print(f"Estimated mu: {mu_mle}")
    print(f"Estimated sigma: {sigma_mle}")
    print(f"Estimated lambda (jump intensity): {lambda_mle}")
    print(f"Estimated mu_j (mean jump size): {mu_j_mle}")
    print(f"Estimated sigma_j (jump volatility): {sigma_j_mle}")

    # Simulate 10 paths based on the MLE results
    S0 = prices[-1]  # Starting price (last available price)
    T = 1  # Time horizon (1 year)
    n_steps = nDays * T
    n_paths = 100  # Number of paths to simulate

    paths = simulate_jump_diffusion_paths(S0, mu_mle, sigma_mle, lambda_mle, mu_j_mle, sigma_j_mle, T, n_steps, n_paths)
    volatilities = calculate_annualized_volatility(paths, dt)

    mean_volatility = np.mean(volatilities)
    std_volatility = np.std(volatilities)

    plt.figure(figsize=(18, 12))
    
    plt.subplot(2,1,1)
    for i in range(n_paths):
        plt.plot(paths[:, i], lw=1.5)
    plt.title(f'Merton Jump Diffusion Model (n={n_paths})')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Price')

    plt.subplot(2,1,2)
    plt.plot(volatilities, lw=2, color='blue')
    plt.axhline(mean_volatility + std_volatility, color='red', linestyle='--', label='+1σ Bound')
    plt.axhline(mean_volatility - std_volatility, color='green', linestyle='--', label='-1σ Bound')
    plt.axhline(mean_volatility, color='gray', linestyle='-', label='Mean Volatility')
    plt.title('Estimated Annualized Vol.')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Annualized Vol.')
    
    fn = os.getenv('USER_HOME','') + '/tmp/merton_jump_model.png'
    plt.savefig(fn)
    print( '-- saved:', fn)

if __name__ == "__main__":
    main()

