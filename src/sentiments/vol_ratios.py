import datetime,os
import pandas as pd
import requests,time
import click
import numpy as np
from multiprocessing import Process


from butil.butils import binance_spot,binance_kline
from strategy.price_disparity import extract_specs 
from ws_bcontract import _main as ws_connector, sync_fetch_ticker

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

def calc_vol( rec, vols=None, contract='' ):
    bid,ask,delta,gamma,theta,vega,impvol,impvolb,impvola = \
        rec['bid'],rec['ask'],rec['delta'],rec['gamma'],rec['theta'],rec['vega'],rec['impvol'],\
            rec['impvol_bid'],rec['impvol_ask']
    if vols:
        rvol_7d = vols['7d']
        rvol_30d = vols['30d']
        rvol_1yr = vols['1yr']
    print(contract, impvol, impvolb,impvola,rvol_7d,rvol_30d,rvol_1yr )

from functools import partial
def _main( contracts, vols ):
    for c in contracts:
        sync_fetch_ticker( c, partial(calc_vol, contract=c, vols = vols,) )

def _mp_main(contracts, vols):
    while True:
        try:
            _main(contracts, vols)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--underlying', default="BTC")
def main(underlying):
    df = fetch_contracts( underlying )
    atm_contracts = get_atm( underlying, df )
    contracts = []
    for expiry, atms in atm_contracts.items():
        for atm in atms:
            contracts += [atm]
            spot_ric, T,K,ctype = extract_specs( atm )
            #print( atm, T, K, ctype )

    # Klines
    ohlcs = binance_kline(f"{underlying.upper()}/USDT", '1d')
    ohlcs.timestamp = ohlcs.timestamp.apply(pd.Timestamp)
    ohlcs.set_index('timestamp', inplace=True, drop=True)
    for n in [1,3,7,14,30]:
        closeNd = ohlcs.resample(f'{n}d').close.agg('last')
        d = closeNd.dropna().pct_change()
        x = 7
        assert d.shape[0]>x, f"No enough data: n={n}, data L={d.shape[0]}"
        sigma = closeNd.dropna().pct_change().rolling(x).apply(np.std).iloc[-1]
        sigma *= np.sqrt(365/n)
        print(f'-- {n}d', f", {(sigma*100):.1f}%" )

    ohlcs['rtn'] = ohlcs.close.pct_change().tail(400)
    vols = {
        "7d":  ohlcs.rtn.rolling(7).apply(np.std).iloc[-1] * np.sqrt(365),
        "30d": ohlcs.rtn.rolling(30).apply(np.std).iloc[-1] * np.sqrt(365),
        "1yr": ohlcs.rtn.rolling(365).apply(np.std).iloc[-1] * np.sqrt(365),
    }

    # Vols
    contracts = contracts[:4] # DEBUG
    conn = Process( target=ws_connector, args=(",".join(contracts), "ticker",) )
    calc = Process( target=_mp_main, args=(contracts, vols, ) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

    #bid,ask = binance_spot(f"{underlying.upper()}/USDT")
    #print('-- spot bid/ask:', bid, ask)
if __name__ == '__main__':
    main()