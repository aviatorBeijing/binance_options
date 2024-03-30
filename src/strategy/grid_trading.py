import pandas as pd 
import click,os

from butil.butils import binance_kline

def _main(df):
    print(df)

@click.command()
@click.option('--ric', default="BTC/USDT")
@click.option('--span', default='1m')
def main(ric,span):
    df = binance_kline(ric, span=span)
    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    df.to_csv(fn)
    print('-- saved:', fn)
    _main(df)

if __name__ == '__main__':
    main()