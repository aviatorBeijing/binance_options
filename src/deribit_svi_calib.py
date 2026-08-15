import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# --- 1. Fetch Live Market Data from Deribit ---
def fetch_live_deribit_options(currency="BTC"):
    url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"
    response = requests.get(url).json()
    raw_data = response['result']
    
    records = []
    now = datetime.utcnow()
    
    for item in raw_data:
        # Instrument format: BTC-29AUG26-65000-C
        parts = item['instrument_name'].split('-')
        expiry_str = parts[1]
        strike = float(parts[2])
        opt_type = parts[3]
        
        # Parse Expiry Date
        expiry_dt = datetime.strptime(expiry_str, "%d%b%y")
        T = (expiry_dt - now).total_seconds() / (365.25 * 86400)
        
        # Skip expired or ultra short-dated (< 12 hours) options
        if T < 0.0015:
            continue
            
        records.append({
            'instrument': item['instrument_name'],
            'expiry': expiry_dt,
            'T': T,
            'strike': strike,
            'type': opt_type,
            'mark_price': item.get('mark_price', 0.0),
            'underlying_price': item.get('underlying_price', 0.0),
            'bid': item.get('bid_price', 0.0) or 0.0,
            'ask': item.get('ask_price', 0.0) or 0.0,
            'mark_iv': item.get('mark_iv', 0.0) / 100.0,  # Convert percentage to decimal
            'volume': item.get('volume', 0.0)
        })
        
    return pd.DataFrame(records)

# --- 2. Live Data Cleaning & Ingestion ---
df_market = fetch_live_deribit_options("BTC")

# Select the nearest major expiry slice with T > 7 days
available_expiries = sorted(df_market['expiry'].unique())
target_expiry = [e for e in available_expiries if (e - datetime.utcnow()).days >= 7][0]

slice_df = df_market[df_market['expiry'] == target_expiry].copy()
T = slice_df['T'].iloc[0]
F = slice_df['underlying_price'].iloc[0]  # Deribit underlying index/forward

# Filter out zero IVs, stale quotes, or quotes with missing bids
clean_df = slice_df[
    (slice_df['mark_iv'] > 0.05) & 
    (slice_df['mark_iv'] < 3.00) &
    (slice_df['bid'] > 0)
].copy()

# Compute Log-Moneyness
clean_df['k'] = np.log(clean_df['strike'] / F)

# Filter log-moneyness range to avoid extreme illiquid tail noise
clean_df = clean_df[(clean_df['k'] >= -0.5) & (clean_df['k'] <= 0.5)]

# --- 3. SVI Calibration Engine ---
def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi_live(df, T):
    k_vec = df['k'].values
    target_w = (df['mark_iv'].values ** 2) * T
    
    # Weight by bid-ask spread proxy
    spread = (df['ask'] - df['bid']).values
    weights = np.where(spread > 0, 1.0 / (spread + 1e-4), 1.0)
    weights /= np.sum(weights)

    def loss(params):
        a, b, rho, m, sigma = params
        if b < 0 or abs(rho) >= 0.99 or sigma <= 0:
            return 1e9
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e9
            
        pred_w = svi_total_variance(k_vec, a, b, rho, m, sigma)
        return np.sum(weights * (pred_w - target_w)**2)

    atm_var = np.median(target_w)
    init_params = [atm_var * 0.8, 0.1, -0.3, 0.0, 0.1]
    
    atm_idx = np.argmin(np.abs(clean_df['k']))
    w_atm = (clean_df['mark_iv'].iloc[atm_idx] ** 2) * T

    init_params = [w_atm * 0.5, 0.1, -0.3, 0.0, 0.1]
    bounds = [(w_atm * 0.05, None), (0.0001, 3.0), (-0.99, 0.99), (-0.5, 0.5), (0.0001, 1.0)]

    res = minimize(loss, init_params, method='L-BFGS-B', bounds=bounds)
    return res.x

# Fit live slice
svi_params = fit_svi_live(clean_df, T)

print(f"Target Expiry: {target_expiry.strftime('%Y-%m-%d')} (T = {T:.4f} years)")
print(f"Underlying Forward: ${F:,.2f}")
print(f"Calibrated SVI Parameters [a, b, rho, m, sigma]:\n{np.round(svi_params, 5)}")
