import pandas as pd 
from tabulate import tabulate 
import numpy as np

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG, get_binance_spot, get_underlying )
from butil.options_calculator import extract_specs,callprice,putprice
from brisk.bfee import calc_fee

def opricer( contracts : list):
    """
    @brief Main purpose of this function is to help 
            placing daily adjusted orders for option contracts.
            Because all Mkt_CALL and Mkt_PUT are the BS conterparts
               shifted by the price differences of the *ask* prices of OTM options, 
               the Mkt_*** prices should be only used as reference for *buy* actions,
               not for sell actions (note "only buy-on-ask" is viable).
    """
    assert len(contracts) == 2, 'Only support a pair for now.' # FIXME
    sbid,sask = get_binance_spot( get_underlying(contracts[0])) # Assume: same underlyings
    smid =(sbid+sask)*.5
    spread = (sbid-sask)/smid
    assert spread < 1/1000, f'Spread is too large: {sbid},{sask},{spread}'
    
    rf = 0. #0.04 # risk-free rate
    dfs = []
    cbids = []
    casks = []

    fairs = [] # Black-Scholes "Fair" Prices of Call/Put
    impvols = {}
    fees = []
    for contract in contracts:
        recs= []
        sym, T, K, ctype = extract_specs(contract)
        cdata = fetch_bidask(contract.upper())
        bid = float(cdata['bid'])
        ask = float(cdata['ask'])
        sigma = float(cdata['impvol_bid'])
        cbids += [bid]
        casks += [ask]
        fees += [ calc_fee(ask,1,contract) + calc_fee(bid,1,contract) ]
        
        if contract not in impvols:
            impvols[contract] = {}
        impvols[contract] = {'bid': float(cdata['impvol_bid']),
                             'ask': float(cdata['impvol_ask'])}

        func_ = None
        if ctype == 'call':
            func_ = callprice
        elif ctype == 'put':
            func_ = putprice
        
        fairs += [ func_(smid,K,T/365,sigma,rf) ]

        mid = (sbid+sask)*.5
        crng = mid*(1. + np.array([0.03, 0.05, 0.08, 0.1, 0.,
                                     -0.03,-0.05,-0.08,-0.1]) )
        for S in crng:
            option_price = func_(S,K,T/365,sigma,rf)
            recs += [ [S,option_price,contract, (bid +ask)*.5 ] ]
        df = pd.DataFrame.from_records( 
                                recs, 
                                columns=[
                                    'Spot',
                                    f'BS_{ctype.upper()}', 
                                    ctype.upper(), 
                                    f'{ctype.upper()}_mid'
                                ]
                            )
        df.set_index('Spot',inplace=True)
        dfs += [df]
    
    df = pd.concat(dfs,axis=1,ignore_index=False)

    dev =  list( np.array(casks)-np.array(fairs )) # a 2 element list
    call_deviation, put_deviation = dev[0],dev[1]
    df['Mkt_CALL'] = df['BS_CALL'] + call_deviation    # shift by the deviation, base on ASK prices of the call/put. 
    df['Mkt_PUT']  = df['BS_PUT'] + put_deviation

    
    df['Spot'] = df.index
    df = df.sort_index(ascending=False)
    df['dp'] = (df['Spot']-sbid)/sbid
    df['c%'] = (df['BS_CALL'] - df['CALL_mid'])/df['CALL_mid']*100
    df['p%'] = (df['BS_PUT'] - df['PUT_mid'])/df['PUT_mid']*100

    df['moneyness'] = df.dp.apply(abs) < 5/1000
    df.moneyness = df.moneyness.apply(lambda s: '*' if s else '')
    df.dp = df.dp.apply(lambda v: f"{(v*100):.1f}%")
    df = df[['moneyness','dp', 'CALL', 'Mkt_CALL', 'BS_CALL','c%','p%','BS_PUT','Mkt_PUT', 'PUT']]
    for col in ['BS_CALL','BS_PUT']:
        df[col] = df[col].apply(lambda v: f'{v:.2f}')
    for col in ['c%','p%']:
        df[col] = df[col].apply(lambda v: f'{v:.1f}')

    print( tabulate(df,headers='keys'))

import click 
@click.command()
@click.option("--contracts", help="comma-separated contracts", default='BTC-250228-95000-C,BTC-250228-95000-P')
def main(contracts):
    """
    Need invoke WSS first, ex.: 
        <USER_HOME>/binance_options/src/ticker.sh BTC-250228-95000-C,BTC-250228-95000-P
    """
    print('-- contracts:',contracts)
    contracts = contracts.upper().split(',')
    opricer( contracts)

if __name__ == '__main__':
    main()