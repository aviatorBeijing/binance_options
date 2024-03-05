import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG


def _main( contracts ):
    try:
        for contract in contracts.split(','):
            with open(f"{DATADIR}/{contract.upper()}.json", 'r') as fh:
                contract_data = json.loads(fh.read())
                print( '\t',contract, '\n', contract_data  )
    except FileNotFoundError as  e:
        print('*** waiting for data ...')
        time.sleep(5)
        return 
    except json.JSONDecodeError as  je:
        print('*** json data conflict, wait ...')
        time.sleep(5)
        return
        
def _mp_main(contracts):
    while True:
        try:
            _main(contracts)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contracts')
def main(contracts):
    conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    calc = Process( target=_mp_main, args=(contracts) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()