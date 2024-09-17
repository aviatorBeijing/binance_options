import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG, get_binance_spot,
                get_binance_next_funding_rate,
                get_maturity, get_underlying )
from brisk.bfee import calc_fee
from ws_bcontract import _main as ws_connector
from strategy.delta_gamma import callprice,putprice
from butil.options_calculator import extract_specs, callprice, putprice, invert_callprice, invert_putprice


def _main(contracts:list, reference_spots:list):
    """
    @brief 
    """
    underlying = get_underlying( contracts[0])
    spot_now, _ = get_binance_spot(underlying)
    recs = [] 
    print('-- ', underlying,": $", spot_now)
    for contract in contracts:
        _, T, K, ctype = extract_specs(contract)
        try:
            cdata = fetch_bidask(contract.upper())
        except Exception as e:
            print('*** wait for data')
            continue
        if not 'ask' in cdata: continue
        ask = float(cdata['ask'])
        sigma = float(cdata['impvol_ask'])

        func_ = None
        if ctype == 'call':
            func_ = callprice
        elif ctype == 'put':
            func_ = putprice
        for S in reference_spots:
            op = func_(S,K,T/365,sigma,0.)
            opr = (op-ask)/ask*100
            spr = (S-spot_now)/spot_now*100
            recs += [ {'contract': contract, 'option_price (ask)':ask, 'spot': f"{S} ({spr:.1f}%)", 'option_projected': op, 'opr': f"{opr:.1f}%"} ]
    df = pd.DataFrame.from_records( recs )

    if not df.empty: df['option_price (ask)'] = df['option_price (ask)'].apply(float)
    
    if not df.empty and 'spot' in df:
        df.sort_values('spot', ascending=True, inplace=True)
        calls = df[df.contract.str.contains('-C')]
        puts  = df[df.contract.str.contains('-P')]

        print()
        print(' '*30, '*** Calls ***')
        print( tabulate(df[df.contract.str.contains('-C')], headers="keys"))
        print(' '*30, '*** Puts ***')
        print( tabulate(df[df.contract.str.contains('-P')], headers="keys"))
        return {
            "ok": True,
            "calls":{
                "columns": [str(s) for s in df.columns ]  if not calls.empty else [],
                "data": [ list(e) for e in calls.to_records(index=False) ] if not calls.empty else [],
            },
            "puts":{
                "columns": [str(s) for s in df.columns ] if not puts.empty else [],
                "data": [ list(e) for e in puts.to_records(index=False) ] if not puts.empty else [],
            }
        }
    else:
        print('*** data error')
        return {
            "ok": False,
            "msg": "wait for market data" 
        }

def _multiprocess_main(contracts:list,projected_spot_prices:list):
    contracts = list( sorted( list(set(contracts)) ) )
    projected_spot_prices = list(set(projected_spot_prices))
    
    print('-- waiting data...')
    time.sleep(2)
    while True:
        try:
            _main( contracts, projected_spot_prices)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contracts', help="contract name")
@click.option('--projected_spot_prices')
@click.option('--adhoc',is_flag=True,default=False)
def main(contracts,projected_spot_prices,adhoc):
    projected_spot_prices = list(map(lambda s: float(s), projected_spot_prices.split(',')))
    projected_spot_prices = list(set(projected_spot_prices))
    contracts = list(set( contracts.split(',')))

    if adhoc:
        _main(contracts,projected_spot_prices)
    else:
        contracts = ','.join(contracts)
        conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
        calc = Process( target=_multiprocess_main, args=(contracts.split(','), projected_spot_prices) )
        conn.start()
        calc.start()
        
        conn.join()
        calc.join()

if __name__ == '__main__':
    main()
