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
    if os.getenv("YAHOO_LOCAL"):
        print("\n","*"*10, " faking prices on local environment","\n")
        bid,ask = 30000,30000 #binance_spot(f"{underlying.upper()}/USDT")
    else:
        bid,ask = binance_spot(f"{underlying.upper()}/USDT")
    df['distance'] = abs(df.strikePrice-(bid+ask)*.5)
    recs = {}
    for expiry in sorted( list(set(df.expiryDate.values))):
        edf = df[df.expiryDate==expiry].sort_values( ['expiryDate','distance'], ascending=True)
        recs[expiry] = list(edf.head(4).symbol.values)
    return recs 

def _dir():
    fdir = os.getenv("USER_HOME", "") + '/tmp/binance_options/'
    fdir += datetime.datetime.strftime(datetime.datetime.today(),'%Y_%m_%d')
    if not os.path.exists(fdir):
        os.makedirs( fdir )
    return fdir 

def refresh_contracts(underlying,update=False):
    fn = f"{_dir()}/_all_binance_contracts_{underlying.lower()}.csv"
    if update:
        df = fetch_contracts( underlying )
        df['expiry'] = df.symbol.apply(lambda s: s.split('-')[1])
        df.to_csv( fn )
    else:
        print('-- reading contracts cached:',fn)
        df = pd.read_csv(fn,index_col=0)
    return df 

def fetch_price_ranges(expiries, odf):
        recs = []
        for datestr in sorted(expiries):
            ddf =  odf[ odf['symbol'].str.contains(datestr) ].sort_values('sumOpenInterestUsd', ascending=False)
            cdf = ddf[ddf['symbol'].str.contains('-C')]
            pdf = ddf[ddf['symbol'].str.contains('-P')]
            cps = [float(s.split('-')[2]) for s in cdf.head(3).symbol.values]
            pps = [float(s.split('-')[2]) for s in pdf.head(3).symbol.values]

            cps_btc = [float(s) for s in cdf.head(3).sumOpenInterest.values]
            pps_btc = [float(s) for s in pdf.head(3).sumOpenInterest.values]

            cps_dollar = [float(s) for s in cdf.head(3).sumOpenInterestUsd.values]
            pps_dollar = [float(s) for s in pdf.head(3).sumOpenInterestUsd.values]
            
            crange = ';'.join(   [ f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps_btc,cps_btc)) ] )
            drange = ';'.join(   [ f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps_dollar,cps_dollar)) ] )
            
            prange = ';'.join(   [ f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps,cps)) ] )
            print( datestr, prange ) 
            
            recs +=[{ "expiry": datestr, "price_range": prange,"oi_qty": crange,"oi_value": drange}]
        rdf = pd.DataFrame.from_records( recs )
        
        return rdf 

def fetch_open_interests(df, underlying, refresh_oi=False):
    expiries = list( set(df.expiry.astype(str).values) )
    odf = pd.DataFrame()
    oi_fn = f"{_dir()}/_all_binance_openinterests_{underlying.lower()}.csv"
    
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
    
    return expiries, odf 

def _wrapper_price_range(underlying, show_atm_contracts=False, update=False):
    df = refresh_contracts( underlying,update=update )
    expiries, odf = fetch_open_interests(df, underlying, refresh_oi=update)
    rdf = fetch_price_ranges( expiries, odf )
    
    rsp = {
        "columns": list(rdf.columns),
        "data": [list(e) for e in rdf.to_records(index=False) ]
    }

    if show_atm_contracts:
        r = get_atm(underlying,df)
        rsp['atm_contracts']={}
        rsp['atm_contracts']['columns'] = list( r.keys())
        rsp['atm_contracts']['data'] = list( r.values())

    return rsp

@click.command()
@click.option('--underlying', default="BTC")
@click.option('--update', is_flag=True, default=False, help='update contracts list')
@click.option('--refresh_oi', is_flag=True, default=False, help='update OI of contracts')
@click.option('--check_price_ranges', is_flag=True, default=False)
def main(underlying,update,refresh_oi, check_price_ranges):
    assert underlying and len(underlying)>0, "Must provide --underlying=<BTC|ETH|etc.>"

    df = refresh_contracts( underlying,update=update )
    expiries, odf = fetch_open_interests(df, underlying, refresh_oi=refresh_oi)

    if check_price_ranges:
        print('\n-- prices range indicated by options OI (implied by insidious market-maker, who can sell options on binance):')
        fetch_price_ranges( expiries, odf )
        import sys;sys.exit()

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

    fn = f"{_dir()}/_atms_{underlying.lower()}.csv"
    with open(fn, 'w') as fh:
        fh.write(','.join(contracts))
    print('-- written:', f"{_dir()}/_all_binance_contracts_{underlying.lower()}.csv")
    print('-- written:', fn )

if __name__ == '__main__':
    main()
