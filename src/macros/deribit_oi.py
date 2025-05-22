import os
import asyncio
import ccxt.pro as ccxtpro

OUTPUT_DIR = og.getenv("USER_HOME","") + "/tmp/deribit"
TOP_N = 10

def get_top_oi_contracts():
    contracts = []

    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith("_dat"):
            try:
                parts = fname.split("_")
                symbol = parts[0]
                oi_str = parts[1]
                oi = float(oi_str)
                contracts.append((symbol, oi))
            except ValueError:
                continue  # Skip malformed files

    top_contracts = sorted(contracts, key=lambda x: x[1], reverse=True)[:TOP_N]
    print(f"\nTop {TOP_N} contracts by Open Interest:")
    for symbol, oi in top_contracts:
        print(f"{symbol} | OI: {oi}")

async def fetch_and_save_oi_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exchange = ccxt.pro.deribit({
        'enableRateLimit': True,
    })

    await exchange.load_markets()

    btc_options = [
        market['symbol']
        for market in exchange.markets.values()
        if market.get('base') == 'BTC' and market.get('option') and not market.get('expired')
    ]

    print(f"Found {len(btc_options)} BTC options")

    for symbol in btc_options:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            oi = ticker.get('info', {}).get('open_interest')

            if oi is not None:
                sanitized_symbol = symbol.replace("/", "")
                filename = f"{sanitized_symbol}_{oi}_dat"
                filepath = os.path.join(OUTPUT_DIR, filename)
                open(filepath, "w").close()
                print(f"Saved: {filepath}")
            else:
                print(f"No OI data for {symbol}")

        except Exception as e:
            print(f"Failed for {symbol}: {e}")

    await exchange.close()

# Run both steps
async def main():
    #await fetch_and_save_oi_files()
    get_top_oi_contracts()

asyncio.run(main())

