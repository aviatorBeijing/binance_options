import os,datetime,click 
import pandas as pd
from butil.butils import binance_kline

def _secs(span):
    span = span.lower()
    if span.endswith('m'):
        n = span.split('m')[0];n=int(n)
        return n * 60
    elif span.endswith('h'):
        n = span.split('h')[0];n=int(n)
        return n * 60 * 60
    elif span.endswith('d'):
        n = span.split('d')[0];n=int(n)
        return n * 60 * 60  * 24
    else:
        raise Exception("Not imp.")

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--span', default='5m')
def main(ric,span):
    df = binance_kline( symbol=ric, span=span )
    for col in ['open','high','low','close','volume']: df[col] = df[col].apply(float)
    row = df.iloc[-1]
    ts,open,high,low,close,volume = row.timestamp,row.open,row.high,row.low,row.close,row.volume 
    ts = datetime.datetime.strptime(ts,'%Y-%m-%dT%H:%M:%S.%fZ')
    tnow = datetime.datetime.utcnow()
    r = (tnow-ts).seconds/_secs(span)*100

    df['oc'] = df['open'] - df['close']

    def _r(x):
        neg = df.oc.iloc[-1]<0
        ocr = x.oc.rolling(x.shape[0]).rank(pct=True).iloc[-1]
        print(f'-- open close rank ({"-" if neg else "+"}): {ocr:.1f}%')
    _r( df[df.oc<0].dropna() )

    print( f"-- kline completeness: {r:.1f}%, close = ${close}" )
    print( f"-- current (UTC): {tnow}")

if __name__ == '__main__':
    main()