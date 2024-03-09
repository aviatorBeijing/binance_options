import datetime,os
import pandas as pd
import requests
import click
import numpy as np
import pandas as pd
from tabulate import tabulate

from butil.butils import binance_spot
from strategy.price_disparity import extract_specs
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
def main(underlying):
    fdir = os.getenv("USER_HOME", "/home/ubuntu") + '/tmp'

    df = fetch_contracts( underlying )
    df.to_csv(f"{fdir}/_all_binance_contracts.csv")

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
    print( tabulate(df, headers="keys") )

    fn = f"{fdir}/_atms.csv"
    with open(fn, 'w') as fh:
        fh.write(','.join(contracts))
    print('-- written:', f"{fdir}/_all_binance_contracts.csv")
    print('-- written:', fn )

if __name__ == '__main__':
    main()