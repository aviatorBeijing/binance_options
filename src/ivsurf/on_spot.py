import pandas as pd 
from tabulate import tabulate 
import numpy as np

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG, get_binance_spot, get_underlying )
from butil.options_calculator import extract_specs,callprice,putprice
from brisk.bfee import calc_fee

def opricer( contracts : list, cap_call: float, cap_put: float):
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

    df['Mkt_CALL'] = df['Mkt_CALL'].apply(lambda v: round(v/5)*5)
    df['Mkt_PUT']  = df['Mkt_PUT'].apply(lambda v: round(v/5)*5)
    
    # capital allocation strategy
    call_cap_limit = cap_call
    put_cap_limit = cap_put
    shares = np.array([1,2,6,18]) # Double the sum of all previous orders
    ttl = np.sum( shares )
    per_c = (call_cap_limit -10.) / ttl 
    per_p = (put_cap_limit -10.) / ttl
    
    df['Qty_CALL'] = 0
    df.loc[df.index[-4:], 'Qty_CALL']  = per_c * shares
    df['Qty_CALL'] /= df['Mkt_CALL']
    df['Qty_CALL'] = df['Qty_CALL'].apply(lambda v: np.floor(v*100)/100.)
    df['Cost_CALL($)'] = df['Qty_CALL'] * df['Mkt_CALL']

    df['Qty_PUT'] = 0
    df.loc[df.index[:4], 'Qty_PUT']  = per_p * shares 
    df['Qty_PUT'] /= df['Mkt_PUT']
    df['Qty_PUT'] = df['Qty_PUT'].apply(lambda v: np.floor(v*100)/100.)
    df['Cost_PUT($)'] = df['Qty_PUT'] * df['Mkt_PUT']
    
    ttl_call = (df['Qty_CALL'] * df['Mkt_CALL']).sum()
    ttl_put  = (df['Qty_PUT'] * df['Mkt_PUT']).sum()
    assert ttl_call <= call_cap_limit, f'Call: {ttl_call} > {call_cap_limit}, insufficient fund (rounding error?).'
    assert ttl_put  <= put_cap_limit, f'Put: {ttl_put} > {put_cap_limit}, insufficient fund (rounding error?).'
    # -end-

    df['Spot'] = df.index
    df.loc[ df['Spot']>smid, 'Mkt_CALL' ] = ''
    df.loc[ df['Spot']<smid, 'Mkt_PUT' ] = ''
    df = df.sort_index(ascending=False)
    df['dp'] = (df['Spot']-sbid)/sbid
    df['c%'] = (df['BS_CALL'] - df['CALL_mid'])/df['CALL_mid']*100
    df['p%'] = (df['BS_PUT'] - df['PUT_mid'])/df['PUT_mid']*100

    df['moneyness'] = df.dp.apply(abs) < 5/1000
    df.moneyness = df.moneyness.apply(lambda s: '*' if s else '')
    df.dp = df.dp.apply(lambda v: f"{(v*100):.1f}%")
    df = df[['moneyness', 'CALL', 'dp','Mkt_CALL', 'Qty_CALL', 'Cost_CALL($)', 'BS_CALL','c%','p%','BS_PUT','Mkt_PUT','Qty_PUT','Cost_PUT($)','dp','PUT']]
    for col in ['BS_CALL','BS_PUT']:
        df[col] = df[col].apply(lambda v: f'{v:.2f}')
    for col in ['c%','p%']:
        df[col] = df[col].apply(lambda v: f'{v:.1f}')

    df.index = list(map(lambda v: f'{v:,.0f}', df.index ))
    new_row = df[['Cost_CALL($)','Cost_PUT($)']].sum()
    df.loc['Ttl'] = new_row
    df = df.fillna('_')

    print( tabulate(df,headers='keys'))

import click 
@click.command()
@click.option("--contracts", help="comma-separated contracts", default='BTC-250228-95000-C,BTC-250228-95000-P')
@click.option('--cap_call', help='capital limit for call options orders', default=500.)
@click.option('--cap_put', help='capital limit for call options orders', default=500.)
def main(contracts,cap_call,cap_put):
    """
    Need invoke WSS first, ex.: 
        <USER_HOME>/binance_options/src/ticker.sh BTC-250228-95000-C,BTC-250228-95000-P
    """
    print('-- contracts:',contracts)
    contracts = contracts.upper().split(',')
    opricer( contracts, cap_call, cap_put)

if __name__ == '__main__':
    main()