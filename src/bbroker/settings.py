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
ex.load_markets()

spot_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'spot',
    }
})
spot_ex.load_markets()