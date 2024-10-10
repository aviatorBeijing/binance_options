import numpy as np
import scipy.stats as si

# Black-Scholes formula for option price
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_option_price(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * si.norm.cdf(d_1) - K * np.exp(-r * T) * si.norm.cdf(d_2)

def put_option_price(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * si.norm.cdf(-d_2) - S * si.norm.cdf(-d_1)

# Vega calculation
def vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * si.norm.pdf(d_1)

# Parameters
S0 = 63000  # Initial price
K = 63500   # Strike price
r = 0.05    # Risk-free rate
T = 4 / 365 # Maturity in years (7 days)

sigmas = [0.2, 0.45, 0.60] # Initial volatility (assumed)
data = []

for sigma in sigmas:
    call_price = call_option_price(S0, K, T, r, sigma)
    put_price = put_option_price(S0, K, T, r, sigma)
    vega_value = vega(S0, K, T, r, sigma)
    data.append([sigma, vega_value, call_price, put_price])

import pandas as pd

df = pd.DataFrame(data, columns=["Sigma", "Vega", "Call Price", "Put Price"])
df['ds'] = df['Sigma'].pct_change().fillna(0).apply(lambda v: f'{(v*100):.1f}%')

df = df[["Sigma","ds", "Vega", "Call Price", "Put Price"] ]
print(df)
