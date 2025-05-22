import os
import ccxt
import requests
import time

OUTPUT_DIR = os.getenv("USER_HOME","") + "/tmp/okex"
TOP_N = 10

okx = ccxt.okx({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'option',  # Ensure we're accessing options market
            }
        })

        # Load market data
okx.load_markets()

import ccxt

def get_okx_option_open_interest(symbol: str) -> dict:
    try:
        # Fetch open interest
        open_interest = okx.fetch_open_interest(symbol)
        return open_interest

    except Exception as e:
        print(f"Error fetching open interest for {symbol}: {e}")
        return {}


def fetch_okex_btc_options_with_oi():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all BTC options
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "OPTION", "uly": "BTC-USD"}
    instruments = requests.get(url, params=params).json().get("data", [])
    print(f"Found {len(instruments)} BTC options on OKX")

    for inst in instruments:
        symbol = inst["instId"]

        try:
            oi_dict = get_okx_option_open_interest(symbol)
            oi_str = oi_dict['openInterestAmount']
            if oi_str:
                oi = float(oi_str)
                filename = f"{symbol}_{oi}_dat"
                filepath = os.path.join(OUTPUT_DIR, filename)
                open(filepath, "w").close()
                print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed: {symbol} | {e}")
            time.sleep(0.2)  # avoid rate limits

def get_top_oi_okex():
    contracts = []
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith("_dat") and "_" in fname:
            try:
                parts = fname.split("_")
                symbol = parts[0]
                oi = float(parts[1])
                contracts.append((symbol, oi))
            except:
                continue

    top = sorted(contracts, key=lambda x: x[1], reverse=True)[:TOP_N]
    print(f"\nTop {TOP_N} OKX BTC Options by Open Interest:")
    for sym, oi in top:
        print(f"{sym} | OI: {oi}")

# Run both
#fetch_okex_btc_options_with_oi()
get_top_oi_okex()

