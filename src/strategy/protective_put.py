import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee


ex = ccxt.binance()

def _v(v): return float(v)
def calc_profits_profile(spot_quantity, contract, cdata):
    bid,ask,vbol, avol, delta = _v(cdata['bid']),_v(cdata['ask']),_v(cdata['bidv']),_v(cdata['askv']),_v(cdata['delta'])
    spot_symbol = contract.split('-')[0]+'/USDT'
    spot_price = ex.fetch_ticker(spot_symbol)['bid']
    strike = contract.split('-')[2];strike = float(strike)

    nominal = 1
    step = 5
    if any( [ e in contract for e in ['BTC-','ETH-'] ]) :
        nominal = 1
        step = 5
    elif any( [ e in contract for e in ['DOGE-',] ]) :
        nominal = 1_000
        step = 0.001

    low = spot_price*0.9
    high = spot_price*1.1
    contract_quantity = spot_quantity/nominal
    premium = ask*contract_quantity
    fee = calc_fee(ask, contract_quantity, contract, is_taker=True)
    cost = premium + fee
    recs = []
    print('-- contract: ', contract)
    print(f'\t premium = ${premium:.2f}')
    print(f'\t     fee = ${fee:.2f}')
    print(f'\t    cost = ${cost:.2f}')
    print(f'\t     qty = {contract_quantity:.2f} contracts @ ${ask:.4f}/contract')
    for price  in np.arange(low,high,step):
        contract_value = max(0, strike-price )*contract_quantity*nominal - cost
        spot_value = price * spot_quantity
        protective_put_value = contract_value + spot_value
        recs += [ [price, protective_put_value, contract_value, spot_value ] ]
    df = pd.DataFrame.from_records( recs )
    df.columns = ['spot','protective','put_value','spot_value']
    print( df )

def _main( contract, spot_quantity ):
    try:
        with open(f"{DATADIR}/{contract.upper()}.json", 'r') as fh:
            contract_data = json.loads(fh.read())
            print( '\t',contract )
    except FileNotFoundError as  e:
        print('*** waiting for data ...')
        time.sleep(5)
        return 
    except json.JSONDecodeError as  je:
        print('*** json data conflict, wait ...')
        time.sleep(5)
        return
    
    calc_profits_profile( spot_quantity, contract, contract_data )

def _mp_main(spot_quantity,contract):
    while True:
        try:
            _main(contract, spot_quantity)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--spot_quantity', default=1.)
@click.option('--contract')
def main(spot_quantity,contract):
    conn = Process( target=ws_connector, args=(f"{contract}", "ticker",) )
    calc = Process( target=_mp_main, args=(spot_quantity,contract) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()