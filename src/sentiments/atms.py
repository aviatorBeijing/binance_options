import datetime,os
import pandas as pd
import requests
import click,time
import numpy as np
import pandas as pd
from tabulate import tabulate

from butil.butils import binance_spot
from strategy.price_disparity import extract_specs

def fetch_oi( expiry, underlying='BTC'):
    try:
        recs = requests.get(url='https://eapi.binance.com/eapi/v1/openInterest',
                params={'underlyingAsset': underlying.upper(), 
                            'expiration':expiry}).json()
        df = pd.DataFrame.from_records( recs )
        return df
    except Exception as e:
        print("*** fetch_oi failed: ", expiry, underlying)
        return pd.DataFrame()

def fetch_contracts(underlying):
    endpoint='https://eapi.binance.com/eapi/v1/exchangeInfo'
    resp = requests.get(endpoint)
    if resp:
        resp  = resp.json()
        ts = resp['serverTime']
        print('-- server time:', pd.Timestamp( datetime.datetime.fromtimestamp( int(float(ts)/1000)) ) )
        rics = resp['optionSymbols']
        df = pd.DataFrame.from_records( rics )
        
        df.expiryDate = df.expiryDate.apply(float).apply(lambda v:v/1000).apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
        df.strikePrice = df.strikePrice.apply(float)
        df['tickSize'] = df['filters'].apply(lambda v: v[0]['tickSize'] )
        #df['minQty'] = df['filters'].apply(lambda v: v[1]['minQty'])
        
        df = df[df.symbol.str.startswith(underlying.upper())]
        df.drop(['filters','contractId','unit','id'], axis=1, inplace=True)
        df = df.sort_values(['expiryDate','symbol','strikePrice'], ascending=True)
        df.reset_index(inplace=True,drop=True)
        return df 
    return pd.DataFrame()

def get_atm( underlying, df ):
    bid,ask = binance_spot(f"{underlying.upper()}/USDT")
    df['distance'] = abs(df.strikePrice-(bid+ask)*.5)
    recs = {}
    for expiry in sorted( list(set(df.expiryDate.values))):
        edf = df[df.expiryDate==expiry].sort_values( ['expiryDate','distance'], ascending=True)
        recs[expiry] = list(edf.head(4).symbol.values)
    return recs 

@click.command()
@click.option('--underlying', default="BTC")
@click.option('--refresh_oi', is_flag=True, default=False)
def main(underlying, refresh_oi):
    assert underlying and len(underlying)>0, "Must provide --underlying=<BTC|ETH|etc.>"

    fdir = os.getenv("USER_HOME", "/home/ubuntu") + '/tmp'

    df = fetch_contracts( underlying )
    df['expiry'] = df.symbol.apply(lambda s: s.split('-')[1])
    df.to_csv(f"{fdir}/_all_binance_contracts_{underlying.lower()}.csv")
    
    # Open Interests
    expiries = list( set(df.expiry.values) )
    odf = pd.DataFrame()
    oi_fn = f"{fdir}/_all_binance_openinterests_{underlying.lower()}.csv"
    if refresh_oi:
        oi_df = []
        for expiry in expiries:
            print('-- expiry:', expiry)
            oi = fetch_oi( expiry, underlying=underlying)
            if not oi.empty:
                oi_df += [ oi ]
            time.sleep(1)
        if oi_df:
            odf = oi_df = pd.concat( oi_df, axis=0)
            oi_df.to_csv(oi_fn, index=False)
    else: 
        if os.path.exists( oi_fn):
            odf = pd.read_csv(oi_fn)
        else:
            print('-- use "--refresh_oi" to cach open interest data first.')
            raise Exception("Empty OI")
   
    odf.sumOpenInterestUsd = odf.sumOpenInterestUsd.apply(float)
    print('-- ranked all by OI:')
    print(tabulate(odf.sort_values('sumOpenInterestUsd', ascending=False).head(5), headers="keys"))

    atm_contracts = get_atm( underlying, df )
    contracts = []
    recs = []
    for expiry, atms in atm_contracts.items():
        for atm in atms:
            contracts += [atm]
            spot_ric, T,K,ctype = extract_specs( atm )
            recs += [ (spot_ric, T,K,ctype, atm,)]
    df = pd.DataFrame.from_records( recs )
    df.columns = 'spot_ric,T,K,ctype,contract'.split(',')

    _f = lambda v: f"$ {v:,.0f}" if not isinstance(v, str) else v
    df['raw_oi'] = df.contract.apply(lambda s: odf[odf.symbol==s].sumOpenInterestUsd.iloc[0])
    df['oi'] = df.raw_oi.apply(lambda s: _f(s))
    
    print('-- ranked ATM by OI:')
    print( tabulate(df.sort_values('raw_oi', ascending=False), headers="keys") ) 
    df.drop(['raw_oi'], inplace=True, axis=1)

    from butil.butils import get_binance_spot
    get_binance_spot()
    print('-- ATM by maturities:')
    print( tabulate(df, headers="keys") )
    print('  -- ATM by maturities (Puts):')
    print( tabulate(df[df.ctype=='put'], headers="keys") )
    print('  -- ATM by maturities (Calls):')
    print( tabulate(df[df.ctype=='call'], headers="keys") )

    fn = f"{fdir}/_atms_{underlying.lower()}.csv"
    with open(fn, 'w') as fh:
        fh.write(','.join(contracts))
    print('-- written:', f"{fdir}/_all_binance_contracts_{underlying.lower()}.csv")
    print('-- written:', fn )

if __name__ == '__main__':
    main()
