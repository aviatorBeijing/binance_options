import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector, sync_fetch_ticker
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee
from strategy.gamma_scalping import EuropeanOption, Asset

def _main( contracts:list ):
    for opt in contracts:
        delta = opt.greeks['delta']
        gamma = opt.greeks['gamma']
        vol = opt.greeks['impvol']

        opt.on_market_move(
            Asset.get_spot_price( opt.get_underlying() )
        )
    
def _mp_main(contracts:str):
    cts = []
    for contract in contracts.split(','):
        opt = EuropeanOption(contract, 1500, 0.01, 1).init()
        cts += [ opt ]

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