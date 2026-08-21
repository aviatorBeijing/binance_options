import ccxt
import os

apikey = os.getenv('BINANCE_MAIN_OPTIONS_APIKEY', None)
secret = os.getenv('BINANCE_MAIN_OPTIONS_SECRET', None)

ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'option',
    }
})
print('-- Loading option markets')
_ = ex.load_markets()

spot_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})
print('-- Loading spot markets')
_ = spot_ex.load_markets()

perp_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})
print('-- Loading perp markets')
_ = perp_ex.load_markets()
