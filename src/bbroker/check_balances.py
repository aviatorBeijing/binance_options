import os
import pandas as pd

from bbroker.settings import ex

def balances() -> pd.DataFrame:
    bal = ex.fetch_balance( params={"type": "option"})
    balType  =  bal['info']['accountType']
    import pprint;pprint.pprint( bal)
    bal = bal['info']['balances']
    bdf  = pd.DataFrame.from_records(bal)
    bdf['free'] = bdf['free'].apply(float)
    bdf['locked'] = bdf['locked'].apply(float)
    bdf['ttl'] =  bdf['free'] +  bdf['locked']
    bdf.sort_values('ttl', ascending=False, inplace=True)
    return bdf[bdf.ttl>0]