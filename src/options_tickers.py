import os,datetime,click
import pandas as pd 
from tabulate import tabulate

from butil.bsql import fetch_bidask as options_bidask

@click.command()
@click.option('--contracts')
def main(contracts):
    recs = []
    for c in contracts.split(','):
        row = options_bidask( c ) 
        recs += [row]
    df = pd.DataFrame.from_records(recs)
    print( tabulate(df,headers='keys'))
    
if __name__ == '__main__':
    main()