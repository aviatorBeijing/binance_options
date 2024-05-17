from email.policy import default
import os,datetime,click 
import pandas as pd 

from butil.butils import get_binance_next_funding_rate

@click.command()
@click.option('--ric', default='BTC/USDT')
def main(ric):
    ric = ric.upper()
    apr, r, ts = get_binance_next_funding_rate(ric)
    print( apr, r, ts)

if __name__ == '__main__':
    main()