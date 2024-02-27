import os,datetime,json
import pandas as pd 
import click
from tabulate import tabulate

from butils import DATADIR

def _v(v): return float(v)
def calc_straddle( ldata,rdata, strike_left,strike_right, vol):
    fee = 5/10000 # Binance: taker 5/10000, maker 2/10000
    lbid,lask,l_bvol, l_avol = _v(ldata['bid']),_v(ldata['ask']),_v(ldata['bidv']),_v(ldata['askv'])
    rbid,rask,r_bvol, r_avol = _v(rdata['bid']),_v(rdata['ask']),_v(rdata['bidv']),_v(rdata['askv'])
    assert lask<rask, "Left leg has to be less than right leg (offer price, a.k.a. ask price)"
    print(f'-- order volumes  (L): {vol}-contract, (R): {vol}-contract')
    print(f'-- strikes (L): ${strike_left}, (R): ${strike_right}')
    recs = []
    
    for stock in range(40000,70000,1000): # at expiration
        gains = max(strike_left - stock,0)
        gains += max( stock - strike_right, 0)
        cost = lask + rask
        profits = gains - cost 
        recs += [ (stock, cost, gains, profits )]
    df = pd.DataFrame.from_records( recs, columns=['spot','cost', 'gain', 'profit'])

    for col in ['cost','profit']:
        df[col] = df[col].apply(lambda e: f"${e:,.0f}")
    print( tabulate(df, headers="keys"))
    

@click.command()
@click.option('--left', help="left leg contract name")
@click.option('--right')
@click.option('--vol', default=1, help="planned order volume, 1=1BTC contract")
def main(left,right, vol):
    ldata = None;rdata = None
    with open(f"{DATADIR}/{left.upper()}.json", 'r') as fh:
        ldata = json.loads(fh.read())
        print( ldata )
    with open(f"{DATADIR}/{right.upper()}.json", 'r') as fh:
        rdata = json.loads(fh.read())
        print( rdata )
    
    strike_left = float(left.split("-")[-2])
    strike_right= float(right.split("-")[-2])

    if not ldata:
        raise Exception(f'*** {left.upper()} contract is not found in cached dir: {DATADIR}')
    if not rdata: 
        raise Exception(f'*** {right.upper()} contract is not found in cached dir: {DATADIR}')
    
    calc_straddle( ldata,rdata, strike_left,strike_right,vol)


if __name__ == '__main__':
    main()