import ccxt,os

apikey = os.getenv('BINANCE_MAIN_OPTIONS_APIKEY', None)
secret = os.getenv('BINANCE_MAIN_OPTIONS_SECRET', None)
ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'option',
    }
})
print('-- Loading markets')
_ = ex.load_markets()

spot_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'spot',
    }
})
_ = spot_ex.load_markets()

perp_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'future',
    }
})
"""
response = exchange.fapiPrivate_post_leverage({
    'symbol': exchange.market_id(symbol),
    'leverage': leverage
})
"""
_ = perp_ex.load_markets()
