import os 
import ccxt

DATADIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/options'
if not os.path.exists( DATADIR):
    try:
        os.makedirs( DATADIR )
    except Exception as e:
        print('*** Make sure set the "USER_HOME" directory for temporary data storage!')

def get_binance_funding_rate(spot_symbol)->float:
    symbol = spot_symbol.replace('/','').replace('-','')    
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    resp = exchange.fapiPublicGetPremiumIndex()
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
    rt = float(r[0]['lastFundingRate'])
    annual = (1+rt)**(365*3)-1 # Every 8 hours
    return annual, rt