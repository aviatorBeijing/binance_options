import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG,
                get_binance_next_funding_rate,
                get_maturity )
from brisk.bfee import calc_fee
from ws_bcontract import _main as ws_connector
from strategy.delta_gamma import callprice,putprice
from butil.options_calculator import extract_specs, invert_callprice,invert_putprice


def _main(contract,user_cost):
    """
    @brief Given the 
                average cost of an option trade, 
                calc the 
                    spot prices v.s. a list of returns% on the option purchased.
    """
    user_cost  = float(user_cost)
    _, T, K, ctype = extract_specs(contract)
    cdata = fetch_bidask(contract.upper())
    bid = float(cdata['bid'])
    sigma = float(cdata['impvol_bid'])
    
    x = bid/user_cost-1
    rtns = np.array([-.1, -.05,.1,.2,.5,1.0, x ])
    rtns = np.array(sorted(rtns))
    pce_ranges = (1+rtns)*user_cost

    spots = []
    options = []
    for target_price in pce_ranges:
        func_ = None
        if ctype == 'call':
            func_ = invert_callprice
        elif ctype == 'put':
            func_ = invert_putprice
        underlying_price = func_(target_price,K,T/365,sigma,0.)
        spots += [ underlying_price ]
        options += [ target_price ]
    df = pd.DataFrame.from_dict({contract: options, 'rtn': rtns, 'spot': spots })
    df['rtn'] = df.rtn.apply(lambda v: f"{(v*100)}%")
    
    print('-- current bid:', bid)
    print( df )


def _multiprocess_main(contract,user_cost):
    print('-- waiting data...')
    time.sleep(2)
    while True:
        try:
            _main(contract,user_cost)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contract', help="contract name")
@click.option('--user_cost', help="average cost of traded option")
def main(contract,user_cost):

    conn = Process( target=ws_connector, args=(f"{contract}", "ticker",) )
    calc = Process( target=_multiprocess_main, args=(contract, user_cost) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()