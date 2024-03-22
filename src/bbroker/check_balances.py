import ccxt,os
import pandas as pd


apikey = os.getenv('BINANCE_SUB01_APIKEY', None)
secret = os.getenv('BINANCE_SUB01_SECRET', None)
ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'option',
    }
})

def balances() -> pd.DataFrame:
    bal = ex.fetch_balance()
    balType  =  bal['info']['accountType']
    bal = bal['info']['balances']
    bdf  = pd.DataFrame.from_records(bal)
    bdf['free'] = bdf['free'].apply(float)
    bdf['locked'] = bdf['locked'].apply(float)
    bdf['ttl'] =  bdf['free'] +  bdf['locked']
    bdf.sort_values('ttl', ascending=False, inplace=True)
    print( bdf[bdf.ttl>0] )
    return bdf