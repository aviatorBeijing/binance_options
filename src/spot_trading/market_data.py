import os,datetime,click 
from tabulate import tabulate

from butil.butils import binance_kline

def _main(ric,span):
    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    df = binance_kline(ric, span=span, grps=5)

    print(tabulate(df.tail(5),headers="keys"))
    rk_last = df.volume.rolling(100).rank(pct=True).iloc[-5:].values
    print(f'-- latest volumes: {(rk_last[0]*100):.1f}%, {(rk_last[1]*100):.1f}%, {(rk_last[2]*100):.1f}%, {(rk_last[3]*100):.1f}%, {(rk_last[4]*100):.1f}%')

    df.to_csv(fn)
    print('-- saved:', fn)

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--rics',default='')
@click.option('--span', default='5m')
def main(ric,rics,span):
    if rics:
        rics = rics.split(',')
    else:
        assert len(ric)>0, 'Empty ric or rics'
        rics = [ric]

    rics = list(map(lambda ric: ric.upper(),rics ))
    
    if len(rics)>1:
        from multiprocessing import Pool
        from functools import partial
        with Pool(5) as pool:
            pool.map( partial(_main, span=span), rics )
            pool.close();pool.join()
    else:
        _main(rics[0], span )


if __name__ == '__main__':
    main()
