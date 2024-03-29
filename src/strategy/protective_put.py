import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np

from ws_bcontract import _main as ws_connector, sync_fetch_ticker
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee


ex = ccxt.binance()

def _v(v): return float(v)
def calc_profits_profile(cdata,spot_quantity=0., contract=""):
    bid,ask,vbol, avol, delta = _v(cdata['bid']),_v(cdata['ask']),_v(cdata['bidv']),_v(cdata['askv']),_v(cdata['delta'])
    spot_symbol = contract.split('-')[0]+'/USDT'
    spot_price = ex.fetch_ticker(spot_symbol)['bid']
    strike = contract.split('-')[2];strike = float(strike)

    nominal = 1
    step = 5
    if any( [ e in contract for e in ['BTC-'] ]) :
        nominal = 1
        step = 500
    if any( [ e in contract for e in ['ETH-'] ]) :
        nominal = 1
        step = 100
    elif any( [ e in contract for e in ['DOGE-',] ]) :
        nominal = 1_000
        step = 0.01

    low = spot_price*0.95
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
    print(f'\t    spot = ${spot_price:.4f}')

    assert strike > spot_price, f"It's meaningless to use a Put in discount w.r.t. spot."

    # Protective put
    for price  in np.arange(low,high,step):
        contract_value = max(0, strike-price )*contract_quantity*nominal - cost
        spot_value = price * spot_quantity
        protective_put_value = contract_value + spot_value
        recs += [ [price, protective_put_value, contract_value, spot_value ] ]

    df = pd.DataFrame.from_records( recs )
    df.columns = ['spot','protective','put_value','spot_only']
    df['breakeven'] = df.protective > df.spot_only

    df['spot_pct'] = (df.spot - spot_price)/spot_price
    df.spot_pct = df.spot_pct.apply(lambda v: f"{(v*100):.1f}%")

    df['portfolio'] = (df.protective-df.spot_only)/df.spot_only
    df['portfolio'] = df.portfolio.apply(lambda v: f"{(v*100):.1f}%")
    print( df )
from functools import partial
def _main( contract, spot_quantity ):
    """try:
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
    """
    sync_fetch_ticker( contract, partial(
        calc_profits_profile, 
            spot_quantity = spot_quantity,
            contract = contract )
    )
    
    #calc_profits_profile( spot_quantity, contract, contract_data )

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