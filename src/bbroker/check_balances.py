import os
import pandas as pd

from bbroker.settings import ex
from butil.butils import get_binance_spot

def valuation( ass ):
    pair = ass.upper()
    if pair != 'USDT':
        pair += '/USDT'
    else: return 1.0
    b,a = get_binance_spot( pair )
    return (b+a)*0.5


def balances() -> pd.DataFrame:
    bal = ex.fetch_balance( params={"type": "option"})
    balType  =  bal['info']['accountType']
    bal = bal['info']['balances']
    bdf  = pd.DataFrame.from_records(bal)
    bdf['free'] = bdf['free'].apply(float)
    bdf['locked'] = bdf['locked'].apply(float)
    bdf['ttl'] =  bdf['free'] +  bdf['locked']
    bdf.sort_values('ttl', ascending=False, inplace=True)
    bdf = bdf[bdf.ttl>0]
    bdf = bdf[bdf.asset != 'ETHW'] 

    bdf['value'] = bdf['asset'].apply(valuation)
    bdf['value'] = bdf['value'] * bdf['ttl']
    bdf.sort_values('value', ascending=False, inplace=True)
    bdf['_val'] = bdf['value'].apply(lambda v: f"$ {v:,.2f}" ) 
    return bdf

if __name__ == '__main__':
    print( balances() )
