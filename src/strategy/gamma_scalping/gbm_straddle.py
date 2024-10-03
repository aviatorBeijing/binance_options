import numpy as np
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--sigma', default=0.4)
@click.option('--mat', default=3)
def main(sigma,mat):
    print('-- sigma:', sigma)
    print('-- maturity days:', mat)

    # Parameters
    S0 = 60_000  # Initial asset price
    K = 60_000   # Strike price (ATM option)
    r = 0.0  # Risk-free rate
    maturity_days = mat
    T = 1.0/365*maturity_days   # Time to maturity (in years)
    sigma = sigma  # Volatility of the underlying asset
    n_simulations = 100  # Number of simulations

    # Simulate terminal asset prices using GBM
    np.random.seed(47)
    Z = np.random.standard_normal(n_simulations)
    Vt = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate the payoff for the straddle strategy
    call_payoff = np.maximum(Vt - K, 0)
    put_payoff = np.maximum(K - Vt, 0)
    straddle_payoff = call_payoff + put_payoff

    # Total cost of the straddle (premium paid for both options)
    # Assuming Black-Scholes pricing, we'll calculate the premium for call and put
    # Black-Scholes formula for call and put options
    def black_scholes_price(S, K, T, r, sigma, option_type="call"):
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price

    call_premium = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    put_premium = black_scholes_price(S0, K, T, r, sigma, option_type="put")
    total_premium_paid = call_premium + put_premium
    print(f'-- premium paid: {total_premium_paid:.2f}')
    print(f'  -- call: {call_premium:.2f}, put: {put_premium:.2f}')

    # P&L for the straddle
    straddle_pnl = straddle_payoff - total_premium_paid

    # Plotting the relation between asset price and the straddle P&L
    plt.figure(figsize=(10, 6))
    plt.scatter(
            #Vt, 
            #(Vt-S0)/S0*100, 
            Vt-S0,
            straddle_pnl, color="blue", alpha=0.5, s=10, label="Simulated P&L")
    plt.axhline(0, color="red", linestyle="--", label="Break-even line")
    plt.title(f"Straddle Strategy P&L vs. Asset Price at Maturity (in {maturity_days} days)")
    plt.xlabel(f"Asset Price at {maturity_days}-Day Maturity (%)")
    plt.ylabel("Straddle P&L")
    plt.legend()
    plt.grid(True)
    import os;fn = os.getenv('USER_HOME','') + '/tmp/gbm_straddle.png'
    plt.savefig( fn )
    print('-- saved:', fn)

if __name__ == '__main__':
    main()

