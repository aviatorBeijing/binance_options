import datetime,os
import pandas as pd
import requests
import click,time
import numpy as np
import pandas as pd
from tabulate import tabulate
from multiprocessing import Process

from butil.butils import binance_spot
from strategy.price_disparity import extract_specs
from sentiments.atms import fetch_oi,fetch_contracts
from ws_bcontract import _main as ws_connector
from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG,
                get_binance_next_funding_rate,
                get_maturity )

def get_contracts_around( strike, df, datestr=None ):
    df = df.copy()
    year = datetime.datetime.today().date().year%100
    if datestr:
        assert len(datestr)==4, f'Wrong format of the datestr, ex. 0401. Found: {datestr}'
        datestr = f'{year}{datestr}'
        df = df[df.symbol.str.contains(datestr)]
    df['distance'] = abs(df.strikePrice-float(strike))
    recs = {}
    print('  -- first 3 pairs of contracts')
    for expiry in sorted( list(set(df.expiryDate.values))):
        edf = df[df.expiryDate==expiry].sort_values( ['expiryDate','distance'], ascending=True)
        recs[expiry] = list(edf.head(6).symbol.values)
    return recs 

def _v(v): return float(v)

def check_market( contracts:str):
    print('-- wait for data ...')
    time.sleep(5)
    contracts = contracts.split(',')
    spot_symbol = contracts[0].split('-')[0]+'/USDT'
    annual, funding_rate, ts = get_binance_next_funding_rate( spot_symbol)

    for i in range(0,3):
        recs = []
        for c in contracts:
            cdata = fetch_bidask(c.upper())
            bid,ask,bvol,avol = _v(cdata['bid']),_v(cdata['ask']),_v(cdata['bidv']),_v(cdata['askv'])
            recs += [(c, bid,ask,bvol,avol)]
        df = pd.DataFrame.from_records(recs, columns=['contract','bid','ask','bid_vol','ask_vol'])
        print('\n')
        print(f'-- funding: {(annual*100):.1f}% ({(funding_rate*10000):.2f}%%)')
        print( tabulate(df, headers="keys"))
        time.sleep(5)

@click.command()
@click.option('--underlying', default="BTC")
@click.option('--strike', default=0., help="The targeted vicinity price of options strikes.")
@click.option('--date4', default='', help="The expiry date of a specific day (optional)")
def main(underlying, strike,date4):
    assert underlying and len(underlying)>0, "Must provide --underlying=<BTC|ETH|etc.>"
    fdir = os.getenv("USER_HOME", "/home/ubuntu") + '/tmp'
    df = fetch_contracts( underlying )
    df['expiry'] = df.symbol.apply(lambda s: s.split('-')[1])
    fn = f"{fdir}/_all_binance_contracts_{underlying.lower()}.csv"
    df.to_csv(fn)
    print('-- written:',fn)

    bid,ask = binance_spot(f"{underlying.upper()}/USDT")
    print(f'-- [spot] bid: {bid}, ask: {ask}')

    cs = get_contracts_around(strike,df,datestr=date4)
    ois=[];recs=[];expDates=[]
    for expiry, contracts in cs.items():
        for c in contracts:
            expDates += [ c.split('-')[1]]
            spot_ric, T,K,ctype = extract_specs( c )
            recs += [ (spot_ric, T,K,ctype, c,)]
    for expiry in list(set(expDates)):
        oi = fetch_oi(expiry,underlying.upper())
        ois += [ oi ]
    odf = pd.concat( ois, axis=0)
    cdf = pd.DataFrame.from_records( recs )
    cdf.columns = 'spot_ric,T,K,ctype,contract'.split(',')

    odf.set_index('symbol', inplace=True, drop=True)
    cdf.set_index('contract', inplace=True, drop=True)
    df = cdf.merge(odf,left_index=True,right_index=True)
    df.sort_values('K', ascending=False, inplace=True)
    print(df)

    contracts = ','.join(list(df.index))
    conn = Process( target=ws_connector, args=(contracts, "ticker",) )
    calc = Process( target=check_market, args=(contracts,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()
