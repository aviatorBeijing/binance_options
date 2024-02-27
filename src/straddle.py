import os,datetime,json
import pandas as pd 
import click

from butils import DATADIR

def _v(v): return float(v)
def calc_straddle( ldata,rdata):
    lbid,lask,l_bvol, l_avol = _v(ldata['bid']),_v(ldata['ask']),_v(ldata['bidv']),_v(ldata['askv'])
    rbid,rask,r_bvol, r_avol = _v(rdata['bid']),_v(rdata['ask']),_v(rdata['bidv']),_v(rdata['askv'])
    print(type(lbid))
    

@click.command()
@click.option('--left', help="left leg contract name")
@click.option('--right')
@click.option('--use_best', is_flag=True, default=False, help='Use best available market price, i.e., buy on the best offer price, instead of mid price.')
def main(left,right, use_best):
    ldata = None;rdata = None
    with open(f"{DATADIR}/{left.upper()}.json", 'r') as fh:
        ldata = json.loads(fh.read())
        print( ldata )
    with open(f"{DATADIR}/{right.upper()}.json", 'r') as fh:
        rdata = json.loads(fh.read())
        print( rdata )
    
    if not ldata:
        raise Exception(f'*** {left.upper()} contract is not found in cached dir: {DATADIR}')
    if not rdata: 
        raise Exception(f'*** {right.upper()} contract is not found in cached dir: {DATADIR}')
    
    calc_straddle( ldata,rdata)


if __name__ == '__main__':
    main()