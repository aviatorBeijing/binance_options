import os,datetime,click 

from butil.butils import binance_kline

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--span', default='5m')
def main(ric,span):
    ric = ric.upper()

    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    df = binance_kline(ric, span=span, grp=5)
    
    df.to_csv(fn)
    print('-- saved:', fn)

if __name__ == '__main__':
    main()