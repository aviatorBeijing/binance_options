import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector, sync_fetch_ticker
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee
from strategy.gamma_scalping import EuropeanOption, Asset, Spot 

TEST = True

def _main( contracts:list, spot_positions:list ):
    for opt in contracts:
        delta_change, addition = opt.on_market_move()
        if addition:
            """delta = opt.greeks['delta']
            gamma = opt.greeks['gamma']
            vol = opt.greeks['impvol']"""
            
            spot_positions += [addition]
            p1 = Asset.get_spot_price( spot_positions[0].ric ) # value positions based on a the same spot price
            #for s in spot_positions:print(s)
            profit = sum([d.value(p1) for d in spot_positions])
            psum = sum([d.delta for d in spot_positions])

            dt = 'SELL' if delta_change>0 else 'BUY ' if delta_change<0 else 'STAY'
            print(f'-- {dt} {opt.underlying} {abs(delta_change)} @ ${p1} scaples (#spots={len(spot_positions)-1}): profit = ${profit:.4f}, spot positions: {psum:.6f}')
            """spot_delta = sum([d.delta for d in spot_positions])
            option_delta = opt.greeks['delta']
            option_pdelta = opt.position_delta
            print(spot_delta, option_delta, option_pdelta, spot_delta + option_pdelta)"""
    
def _mp_main(contracts:str):
    spot_positions = []

    cts = []
    qty = 1.
    entry = 1_500
    nominal = 1.
    for contract in contracts.split(','):
        opt = EuropeanOption(contract, entry, qty, nominal).init()
        cts += [ opt ]
    underlying = cts[0].underlying
    init_spot = cts[0].init_spot
    td = total_option_deltas = sum( [o.greeks['delta'] for o in cts] )
    for o in cts:
        bid,ask,last_trade = Asset.get_options_price(o.contract)
        print('  -- ', o.contract, bid, ask, last_trade)
    print('-- initial spot:', init_spot)
    print('-- initial option delta:', total_option_deltas)
    print(f'-- upfront {"SHORT" if td>0 else "LONG" if td<0 else "STAY"} {abs(td)*nominal} share of {underlying}')
    spot_positions += [ Spot(underlying,init_spot,total_option_deltas*(-1)) ]

    if TEST:
        i = 0
        for i in range(0,1000):
            try:
                _main(cts, spot_positions)
                test_setting(i);i+=1
            except KeyboardInterrupt as e:
                print("-- Exiting --")
                break
    else:
        while True:
            try:
                _main(cts, spot_positions)
                time.sleep(5)
            except KeyboardInterrupt as e:
                print("-- Exiting --")
                break

def test_setting(i:int):
    fn =os.getenv('USER_HOME','')+f'/tmp/btc-usdt_1h.csv'
    df = pd.read_csv( fn )
    closes = df.close.values

    deltas = [0.48] * len(closes)
    spot = closes
    gammas = [0.0003] * len(closes)

    j = (i+1)%len(closes)
    Asset.get_options_greeks = lambda e: {'delta': deltas[j], 'gamma': gammas[j]}
    Asset.get_spot_price = lambda e: spot[j]
    Asset.get_options_price = lambda e: (1200.00,1800.00, 650.00,)

@click.command()
@click.option('--contracts')
def main(contracts):
    """conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    calc = Process( target=_mp_main, args=(contracts,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() """

    if TEST:
        test_setting(0)
    _mp_main( contracts )

if __name__ == '__main__':
    main()