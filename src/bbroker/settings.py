import ccxt, os

apikey = os.getenv('BINANCE_MAIN_OPTIONS_APIKEY', None)
secret = os.getenv('BINANCE_MAIN_OPTIONS_SECRET', None)

# Base client instances without eager load_markets() calls
ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'option',
    }
})

spot_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

perp_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})

def ensure_markets(exchange_obj):
    """Lazy loader: loads markets only when explicitly required by order routing logic."""
    if not exchange_obj.markets:
        print(f'-- Lazy loading markets for {exchange_obj.id} ({exchange_obj.options.get("defaultType")})')
        exchange_obj.load_markets()
    return exchange_obj
