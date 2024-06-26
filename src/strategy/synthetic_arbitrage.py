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



def _multiprocess_main(contracts:list):
    
    while True:
        try:
            recs = []
            for c in contracts:
                spot_symbol, T,K, ctype = extract_specs( c+'C' )
                T = get_maturity(c+'C')/365
                _,S = binance_spot(spot_symbol)
                call = fetch_bidask( c+'C' )
                C = float(call['ask']) + float(call['bid']); C*=.5
                put = fetch_bidask( c+'P' )
                P = float(put['ask']) + float(put['bid']); P*=.5
                recs += [ {'contract': c[:-1],'call': C, 'put': P, 'spot': S, 'strike': K, 'maturity': T}]
            df = pd.DataFrame.from_records( recs )
            
            df['C+PV'] = df['call'] + df['strike']
            df['P+S']  = df['put'] + df['spot']
            df['implied_rate'] = -np.log( (df['P+S'] - df['call'])/df['strike'] )/df['maturity'];df.implied_rate=df.implied_rate.apply(lambda v: f"{(v*100):.1f}%")
            df['disparity'] = ((df['C+PV'] - df['P+S'])/df['P+S']*100);df.disparity = df.disparity.apply(lambda v: f"{v:.2f}%")

            print( tabulate(df,headers='keys'))
        except AssertionError as e:
            print('waiting for data:', contracts)
        time.sleep(5)


@click.command()
@click.option('--contracts')
def main(contracts):
    cs = []
    for c in contracts.split(','):
        cs += [c[:-1]]
    cs = list(set(cs))
    contracts = []
    for c in cs:
        contracts += [c+'C']
        contracts += [c+'P']
       
    conn = Process( target=ws_connector, args=(','.join(contracts), "ticker",) )
    calc = Process( target=_multiprocess_main, args=(cs,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()