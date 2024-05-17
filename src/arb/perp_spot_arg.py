import os,datetime,click 
import pandas as pd 

from butil.butils import get_binance_next_funding_rate

def main():
    ric = 'ONT/USDT'
    apr, r, ts = get_binance_next_funding_rate(ric)
    print( apr, r, ts)

if __name__ == '__main__':
    main()