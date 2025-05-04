import ccxt,os
import decimal

# Initialize Binance Futures (USD-M)
exchange = ccxt.binance({
    'options': {
        'defaultType': 'future',
    }
})

# Load all markets
markets = exchange.load_markets()

# Prepare result storage
price_digits_map = {}
lines = []
# Iterate through all futures symbols
for symbol, market in markets.items():
    if market['contract'] and market['active'] and market['id'].endswith('USDT'):
        # Extract tick size from PRICE_FILTER
        try:
            price_filter = next(f for f in market['info']['filters'] if f['filterType'] == 'PRICE_FILTER')
            tick_size = float(price_filter['tickSize'])
            effective_digits = abs(decimal.Decimal(str(tick_size)).as_tuple().exponent)
            price_digits_map[symbol] = effective_digits
            quote_asset = market['quote']

            # Format line
            line = f"{symbol},{effective_digits}"
            lines.append(line)

        except Exception as e:
            price_digits_map[symbol] = f"Error: {e}"

# Sort and print
#for symbol in sorted(price_digits_map):
#    print(f"{symbol}: {price_digits_map[symbol]} effective price digits")

# Save to csv
output_file = os.getenv("USER_HOME", "") + "/tmp/binance_perp_ndigits.csv"
# Write to file using raw I/O
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Saved {len(lines)} entries to {output_file}")
