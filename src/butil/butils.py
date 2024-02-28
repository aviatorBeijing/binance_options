import os,datetime
import ccxt

DEBUG = os.getenv("BINANCE_DEBUG", None)

DATADIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/options'
if not os.path.exists( DATADIR):
    try:
        os.makedirs( DATADIR )
    except Exception as e:
        print('*** Make sure set the "USER_HOME" directory for temporary data storage!')

ex_binance = ccxt.binance({'enableRateLimit': True})
ex_bmex = ccxt.bitmex({'enableRateLimit': True})
ex_bybit = ccxt.bybit({'enableRateLimit': True})

def get_bmex_next_funding_rate(spot_symbol)->tuple: 
    if 'BTC' in spot_symbol: spot_symbol.replace('BTC','XBT')
    if 'USDT' in spot_symbol: spot_symbol.replace('USDT','USD')
    symbol = spot_symbol.replace('/','').replace('-','') 
    resp = ex_bmex.fetchFundingRate(symbol)
    rt = float(resp['fundingRate'])
    ts = resp['fundingDatetime'] 
    annual = (1+rt)**(365*3)-1 # Every 8 hours
    return annual, rt, ts

def get_binance_next_funding_rate(spot_symbol)->tuple:
    symbol = spot_symbol.replace('/','').replace('-','')    
    
    resp = ex_binance.fapiPublicGetPremiumIndex()
    r = list(filter(lambda e: symbol == e['symbol'],resp ) )
    assert len(r)==1, f"Funding rate of related perpetual not found: {spot_symbol}"
    '''
    {'estimatedSettlePrice': '396.73116476',
    'indexPrice': '396.03627621',
    'interestRate': '0.00000000',
    'lastFundingRate': '-0.00050799',
    'markPrice': '395.61000000',
    'nextFundingTime': '1709107200000',
    'symbol': 'BNBUSDT',
    'time': '1709090146000'}
    '''
    ts = r[0]['nextFundingTime'] # int in str format
    ts = datetime.datetime.fromtimestamp(int(ts)/1000).isoformat() # str
    rt = float(r[0]['lastFundingRate']) # Binance mis-interperated the "last*" as the "next*"
    annual = (1+rt)**(365*3)-1 # Every 8 hours
    return annual, rt, ts