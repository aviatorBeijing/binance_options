import os,click,talib
import pandas as pd
from tabulate import tabulate

from butil.butils import binance_kline

def _main(sym,wd,dt,prev):
    df = binance_kline(f'{sym.upper()}/USDT', span=dt, grps=5)
    volume = df.volume
    df.volume = talib.EMA(volume, timeperiod=5)
    df['volrank'] = volume.dropna().rolling(wd).rank(pct=True)
    df['rtn'] = df.close.pct_change()
    df['rtnrank'] = df.rtn.rolling(wd).rank(pct=True)
    df['dir'] = df.close > df.open

    df['dir'] = df['dir'].apply(lambda d: '+' if d else '-')
    df.volrank = df.volrank.apply(lambda v: f"{(v*100):.1f}%")
    df.rtnrank = df.rtnrank.apply(lambda v: f"{(v*100):.1f}%")
    #print(df.tail(5) )
    #print(f'-- {sym.upper()}/USDT activity based on dt={dt}, window={wd} (in unit of "dt")')
    x = -1 if not prev else -2
    return {
            'ric': sym.upper(),
            'price': df.iloc[x].close,
            'ts':  df.iloc[x].timestamp,
            'volrank': df.iloc[x].volrank,
            'direction': df.iloc[x]['dir'],
            'daily rtn': f'{(df.iloc[x].rtn*100):.1f}%',
            'rtnrank': df.iloc[x].rtnrank,
            'wd': wd,
            'span': dt,
            }

@click.command()
@click.option('--sym',default='')
@click.option('--syms',default='')
@click.option('--wd', default=7)
@click.option('--dt', default='1d')
@click.option('--prev',is_flag=True,default=False)
def main(sym,syms,wd,dt, prev):
    if syms:
        syms = syms.split(',')
    else:
        assert len(sym)>0, 'Empty symbol: --sym or --syms'
        syms = [sym]
        
    if len(syms)>1:
        from multiprocessing import Pool
        from functools import partial
        with Pool(5) as pool:
            recs = pool.map(
                    partial(_main, wd=wd,dt=dt,prev=prev),
                    syms
                    )
            pool.close();pool.join()
    else:
        recs = _main(syms[0], wd,dt, prev)

    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame.from_records( recs )
    df['_r'] = df.volrank.apply(lambda s: float(s.replace("%","")) )
    df.sort_values('_r', ascending=True, inplace=True)
    df.drop(['_r'], inplace=True, axis=1)
    print(tabulate(df,headers='keys') )

if __name__ == '__main__':
    main()
