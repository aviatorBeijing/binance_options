import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool

plt.style.use('fivethirtyeight')

# Black-Scholes formula (unchanged)
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type")

# Dynamic delta calculation
def delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1

# SABR Model Simulation (with dynamic volatility)
def simulate_sabr(S0, kappa, alpha, rho, nu, T, steps):
    dt = T / steps
    prices = np.zeros(steps + 1)
    volatility = np.zeros(steps + 1)
    prices[0] = S0
    volatility[0] = alpha  # initial volatility assumption
    
    for t in range(1, steps + 1):
        dW_1 = np.random.normal(0, np.sqrt(dt))  # Stock price Brownian motion
        dW_2 = np.random.normal(0, np.sqrt(dt))  # Volatility Brownian motion
        
        # Correlated Brownian motions
        dW_2 = rho * dW_1 + np.sqrt(1 - rho**2) * dW_2
        
        # Volatility process (SABR)
        volatility[t] = volatility[t - 1] + nu * volatility[t - 1] * dW_2
        
        # Asset price process (SABR)
        prices[t] = prices[t - 1] * np.exp((0 - 0.5 * volatility[t]**2) * dt + volatility[t] * prices[t - 1]**alpha * dW_1)
        
        # Make sure volatility stays positive
        volatility[t] = max(volatility[t], 0)
    
    return prices, volatility

# Option pricing and dynamic delta with SABR implied volatility
def simulate_single_path(args):
    S0, Kc, Kp, r, kappa, alpha, rho, nu, T, steps = args
    stock_prices, volatility = simulate_sabr(S0, kappa, alpha, rho, nu, T, steps)
    
    short_strangle_premium = black_scholes(S0, Kc, T, r, volatility[0], "call") + black_scholes(S0, Kp, T, r, volatility[0], "put")
    pnl_path = np.zeros(steps)
    strangle_pnl_path = np.zeros(steps)
    call_prices = np.zeros(steps)
    put_prices = np.zeros(steps)
    call_deltas = np.zeros(steps)
    put_deltas = np.zeros(steps)
    num_shares = 1  # Holding one unit of the underlying stock
    
    # Starting stock position for the covered strangle
    stock_position = num_shares * S0  # Assume buying stock initially at S0
    
    for t in range(steps):
        St = stock_prices[t]
        T_remain = T - t * (T / steps)
        if T_remain <= 0:
            break
        
        # Calculate the option prices and deltas
        sigma_c = volatility[t]
        sigma_p = volatility[t]
        
        call_price = black_scholes(St, Kc, T_remain, r, sigma_c, "call")
        put_price = black_scholes(St, Kp, T_remain, r, sigma_p, "put")
        
        call_delta = delta(St, Kc, T_remain, r, sigma_c, "call")
        put_delta = delta(St, Kp, T_remain, r, sigma_p, "put")
        
        call_prices[t] = call_price
        put_prices[t] = put_price
        call_deltas[t] = call_delta
        put_deltas[t] = put_delta
        
        # Profit and loss for the stock and options
        pnl_stock = num_shares * (St - S0)  # P&L from the underlying stock price movement
        
        # Adjust the portfolio exposure using delta: hedging the option position
        call_position_change = call_delta * (St - stock_prices[t - 1] if t > 0 else 0)
        put_position_change = put_delta * (St - stock_prices[t - 1] if t > 0 else 0)
        
        # Calculate P&L of the options by taking the difference in their prices and delta hedging
        pnl_options = short_strangle_premium - (call_price + put_price) + call_position_change + put_position_change
        pnl_path[t] = pnl_options + pnl_stock
        strangle_pnl_path[t] = short_strangle_premium - (call_price + put_price)
    
    return stock_prices, pnl_path, strangle_pnl_path, call_prices, put_prices, call_deltas, put_deltas

def simulate_covered_strangle_parallel(S0, Kc, Kp, r, kappa, alpha, rho, nu, T, steps, paths):
    args = [(S0, Kc, Kp, r, kappa, alpha, rho, nu, T, steps) for _ in range(paths)]
    with Pool() as pool:
        results = pool.map(simulate_single_path, args)
    stock_prices = np.array([res[0] for res in results]).T
    pnl_paths = np.array([res[1] for res in results]).T
    strangle_pnl_path = np.array([res[2] for res in results]).T
    call_prices = np.array([res[3] for res in results]).T
    put_prices = np.array([res[4] for res in results]).T
    call_deltas = np.array([res[5] for res in results]).T
    put_deltas = np.array([res[6] for res in results]).T
    return stock_prices, pnl_paths, strangle_pnl_path, call_prices, put_prices, call_deltas, put_deltas

# Example parameters for the SABR model
kappa = 2.0   # Mean reversion speed
alpha = 0.2   # Initial volatility level
rho = -0.5    # Correlation between stock and volatility
nu = 0.3      # Volatility of volatility

# Running the simulation with the new SABR model
if __name__ == "__main__":
    S0 = 100
    Kc = 105
    Kp = 95
    r = 0.02
    mu = 0.05
    T = 0.25
    steps = 60
    paths = 100

    stock_prices, pnl_paths, strangle_pnl_path, call_prices, put_prices, call_deltas, put_deltas = simulate_covered_strangle_parallel(
        S0, Kc, Kp, r, kappa, alpha, rho, nu, T, steps, paths
    )

    fig, ax = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = ax[0, 0]
    ax1.plot(pnl_paths[:, -1], label="P&L of Last Path", color='blue')
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Profit & Loss ($)", color='blue')
    ax1.set_title("Time-Dependent P&L ($)")
    ax1.legend(loc='upper left')

    ax2 = ax[0, 1]
    ax22 = ax2.twinx()
    ax22.plot(stock_prices[:, -1], label="Spot Price", color='green', linestyle='solid')
    ax2.plot(call_prices[:, -1], label="Call Option Price", color='red', linestyle='dashed')
    ax2.plot(put_prices[:, -1], label="Put Option Price", color='purple', linestyle='dashdot')
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Options ($)")
    ax22.set_ylabel("Underlying ($)")
    ax2.set_title("Underlying & Option Prices over Time")
    ax2.legend(loc='upper left')
    ax22.legend(loc='upper right')

    ax3 = ax[1, 0]
    ax3.hist(pnl_paths[-1], bins=10, edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Final P&L ($)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Histogram of Final P&L for {paths} Paths")

    ax4 = ax[1, 1]
    final_spot_prices = stock_prices[-1, :]
    final_pnls = pnl_paths[-1, :]
    sfinal_pnls = strangle_pnl_path[-1,:]
    ax4.scatter(final_spot_prices,  final_pnls, c='orange', alpha=0.7)
    ax4.scatter(final_spot_prices, sfinal_pnls, c='gray', alpha=0.9)
    ax4.set_xlabel("Final Underlying Spot Price ($)")
    ax4.set_ylabel("Final P&L ($)")
    ax4.set_title(
            "Final P&L vs Underlying Spot Price"
            "\n(Dynamic Delta)"
        )

    plt.tight_layout()
    plt.savefig('covered_straddle_pnl_sabr_vol.png')

    print('-- Uncalibrated SABR model for demo purpose!!!')

