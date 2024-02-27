import os,datetime,json
import pandas as pd 
import click,time
from tabulate import tabulate
import ccxt

from butils import DATADIR

spot_symbol = 'BTC/USDT'
ex = ccxt.binance()

def _v(v): return float(v)
def calc_straddle( ldata,rdata, strike_left,strike_right, vol):
    fee_rate = 5/10000 # Binance: taker 5/10000, maker 2/10000
    lbid,lask,l_bvol, l_avol = _v(ldata['bid']),_v(ldata['ask']),_v(ldata['bidv']),_v(ldata['askv'])
    rbid,rask,r_bvol, r_avol = _v(rdata['bid']),_v(rdata['ask']),_v(rdata['bidv']),_v(rdata['askv'])
    assert lask<rask, "Left leg has to be less than right leg (offer price, a.k.a. ask price)"
    print(f'-- order volumes  (L): {vol}-contract, (R): {vol}-contract')
    recs = []
    

    premium = (lask + rask)*vol
    adhoc = ex.fetch_ticker(spot_symbol)['bid']
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    fee = vol * adhoc * fee_rate # Binance calc the fee from contract nominal $value.

    for stock in range(40000,70000,1000): # at expiration
        gains = max(strike_left - stock,0)
        gains += max( stock - strike_right, 0)
        gains *= vol
        profits = gains - premium - fee
        recs += [ ( stock, gains, profits )]
    
    df = pd.DataFrame.from_records( recs, columns=[ f"{spot_symbol} @ expiry",'gain', 'net profit @ expiry'])
    cost = premium + fee
    df['stradle_return'] = ( df['net profit @ expiry']) / cost
    df['spot_return'] = (df[f"{spot_symbol} @ expiry"] - adhoc)/adhoc

    for col in ['net profit @ expiry']:
        df[col] = df[col].apply(lambda e: f"${e:,.2f}")
    df['stradle_return'] = df['stradle_return'].apply(lambda v: f"{(v*100):.2f}%")
    df['spot_return'] = df['spot_return'].apply(lambda v: f"{(v*100):.1f}%")
    #df.set_index(['spot_return'],inplace=True,drop=True)
    print( tabulate(df, headers="keys"))
    
    print(f'-- spot: ${adhoc:,.2f} @ {ts} (UTC+8)')
    print(f'-- (assumed) fee_rate: {(fee_rate*100):.2f}%')
    
    print(f'-- order size: {vol} contract  (call&put each)')
    print(f'-- investment  ${premium:,.2f} (premium) + ${fee:,.2f} (fee)')
    

def _main(left,right, vol):
    ldata = None;rdata = None
    print("-- Contracts --")
    try:
        with open(f"{DATADIR}/{left.upper()}.json", 'r') as fh:
            ldata = json.loads(fh.read())
            print( '\t',left, ldata )
        with open(f"{DATADIR}/{right.upper()}.json", 'r') as fh:
            rdata = json.loads(fh.read())
            print( '\t',right, rdata )
    except FileNotFoundError as  e:
        print('*** waiting for data ...')
        time.sleep(5)
        return 
    except json.JSONDecodeError as  je:
        print('*** json data conflict, wait ...')
        time.sleep(5)
        return 
    
    if not ldata:
        raise Exception(f'*** {left.upper()} contract is not found in cached dir: {DATADIR}')
    if not rdata: 
        raise Exception(f'*** {right.upper()} contract is not found in cached dir: {DATADIR}')
    
    strike_left = float(left.split("-")[-2])
    strike_right= float(right.split("-")[-2])
    calc_straddle( ldata,rdata, strike_left,strike_right,vol)

from multiprocessing import Process
from ws_bcontract import _main as ws_connector

def _multiprocess_main(left,right,vol):
    while True:
        try:
            _main(left,right,vol)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--left', help="left leg (OTM put option) contract name")
@click.option('--right', help="right leg (OTM call option)")
@click.option('--size', default=1.0, help="1, 0.1, ... Contract size, 1=1BTC contract")
def main(left,right, size):

    conn = Process( target=ws_connector, args=(f"{left},{right}", "ticker",) )
    calc = Process( target=_multiprocess_main, args=(left,right,size,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()


if __name__ == '__main__':
    main()