import os,datetime,json
import pandas as pd 
import click

from butils import DATADIR

@click.command()
@click.option('--left', help="left leg contract name")
@click.option('--right')
@click.option('--use_best', is_flag=True, default=False, help='Use best available market price, i.e., buy on the best offer price, instead of trade price.')
def main(left,right, use_best):
    ldata = None;rdata = None
    with open(f"{DATADIR}/{left.upper()}.json", 'r') as fh:
        ldata = json.loads(fh.read())
        print( ldata )
    with open(f"{DATADIR}/{right.upper()}.json", 'r') as fh:
        rdata = json.loads(fh.read())
        print( rdata )

if __name__ == '__main__':
    main()