import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG, binance_spot, bjnow_str,
                get_binance_next_funding_rate, get_binance_spot,
                get_maturity, get_underlying )
from brisk.bfee import calc_fee
from ws_bcontract import _main as ws_connector
from butil.options_calculator import extract_specs, invert_callprice,invert_putprice,callprice,putprice


def _main(contract,user_cost, reference_spot):
    """
    @brief Given the 
                average cost of an option trade, 
                calc the 
                    spot prices v.s. a list of returns% on the option purchased.
    @param reference_spot: the price of underlying when the routine starts, will be not update in the middle!
    """
    user_cost  = float(user_cost)
    _, T, K, ctype = extract_specs(contract)
    cdata = fetch_bidask(contract.upper())
    bid = float(cdata['bid'])
    sigma = float(cdata['impvol_bid'])
    
    x = bid/user_cost-1
    rtns = np.array([-.1, -.05,.1,.2,.5,1.0, 2.0, 3.0,4.0,5.0, x ])
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
    df['spot_chg'] = (df.spot - reference_spot)/reference_spot*100
    df['spot_chg'] = df['spot_chg'].apply(lambda v: f"{v:.1f}%")
    
    g = bid-user_cost
    print('-- current bid:', bid, ', cost:', user_cost, f', ${"+" if g>0 else ""}{g}')
    print('-- price projections of option v.s. spot (assuming impvol is NOT changing)')
    print( df )


def _multiprocess_main(contract,user_cost):
    print('-- waiting data...')
    bid,ask = binance_spot( get_underlying(contract))
    spread = (bid-ask)/(bid+ask)*2
    assert spread < 1/1000, f'Spread is too large. {contract}, {bid},{ask},{spread}'

    time.sleep(2)
    while True:
        try:
            _main(contract,user_cost, (bid+ask)*.5)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

def _multicontracts_loop(contracts:list):
    print('waiting for options data...')
    time.sleep(5)
    while True:
        _multicontracts_main(contracts)
        print( bjnow_str(), ', next calc in 10 secs','\n' )
        time.sleep(10)

def _multicontracts_main(contracts:list):
    sbid,sask = get_binance_spot( get_underlying(contracts[0]))
    spread = (sbid-sask)/(sbid+sask)*2
    assert spread < 1/1000, f'Spread is too large. {contracts[0]}, {sbid},{sask},{spread}'
    
    dfs = []
    for contract in contracts:
        recs= []
        sym, T, K, ctype = extract_specs(contract)
        cdata = fetch_bidask(contract.upper())
        bid = float(cdata['bid'])
        ask = float(cdata['ask'])
        sigma = float(cdata['impvol_bid'])

        func_ = None
        if ctype == 'call':
            func_ = callprice
        elif ctype == 'put':
            func_ = putprice
        for S in np.arange( sbid*(1-0.1), sbid*(1+0.1), 200):
            option_price = func_(S,K,T/365,sigma,0.)
            recs += [ [S,option_price,contract] ]
        df = pd.DataFrame.from_records( recs, columns=['price',f'BS_{ctype.upper()}', ctype.upper()] )
        df.set_index('price',inplace=True)
        dfs += [df]

    df = pd.concat(dfs,axis=1,ignore_index=False)
    df['price'] = df.index
    df['dp'] = (df.price-sbid).apply(abs)/sbid
    df['moneyness'] = df.dp < 1./100
    df.moneyness = df.moneyness.apply(lambda s: '*' if s else '')
    df.dp = df.dp.apply(lambda v: f"{(v*100):.1f}%")
    df = df[['CALL','BS_CALL','dp','BS_PUT','PUT','moneyness']]
    print( tabulate(df,headers='keys'))

@click.command()
@click.option('--contract', help="contract name")
@click.option('--user_cost', help="average cost of traded option")
@click.option('--contracts', help="multiple contracts",default="")
def main(contract,user_cost,contracts):

    if not contracts and contract:
        calc = Process( target=_multiprocess_main, args=(contract, user_cost) )
    else:
        calc = Process( target=_multicontracts_loop, args=(contracts.split(','),) )
    
    if not contracts:
        assert contract, 'Must provide contract'
        contracts = contract
    conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()
