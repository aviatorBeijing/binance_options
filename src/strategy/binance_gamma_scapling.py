import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector, sync_fetch_ticker
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee
from strategy.gamma_scalping import EuropeanOption, Asset, Spot 

spot_positions = [] # TODO Need a storage method, to track the scraple history. FIXME

def _main( contracts:list ):
    global spot_positions
    for opt in contracts:
        delta = opt.greeks['delta']
        gamma = opt.greeks['gamma']
        vol = opt.greeks['impvol']

        _, addition = opt.on_market_move()
        if addition:
            spot_positions += [addition]
            p1 = Asset.get_spot_price( spot_positions[0].ric ) # value positions based on a the same spot price
            profit = sum([d.value(p1) for d in spot_positions[1:]])
            psum = sum([d.delta for d in spot_positions[1:]])

            print(f'    -- scaples (#{len(spot_positions)-1}): ${profit:.4f}, spot positions: {psum}')
            
            """spot_delta = sum([d.delta for d in spot_positions])
            option_delta = opt.greeks['delta']
            option_pdelta = opt.position_delta
            print(spot_delta, option_delta, option_pdelta, spot_delta + option_pdelta)"""
    
def _mp_main(contracts:str):
    global spot_positions

    cts = []
    qty = 1.
    nominal = 1.
    for contract in contracts.split(','):
        opt = EuropeanOption(contract, 1500, qty, nominal).init()
        cts += [ opt ]
    underlying = cts[0].underlying
    init_spot = cts[0].init_spot
    td = total_option_deltas = sum( [o.greeks['delta'] for o in cts] )
    for o in cts:
        bid,ask,last_trade = o.get_options_price(o.contract)
        print('  -- ', o.contract, bid, ask, last_trade)
    print('-- initial spot:', init_spot)
    print('-- initial option delta:', total_option_deltas)
    print(f'-- upfront {"SHORT" if td>0 else "LONG" if td<0 else "STAY"} {abs(td)*nominal} share of {underlying}')
    spot_positions += [ Spot(underlying,init_spot,total_option_deltas*(-1)) ]

    while True:
        try:
            _main(cts)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contracts')
def main(contracts):
    conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    calc = Process( target=_mp_main, args=(contracts,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()