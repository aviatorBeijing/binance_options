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

    df['oc'] = df['close'] - df['open']
    df['hl'] = df['high'] - df['low']

    print(f'-- span={span}, n={df.shape[0]}')
    print( f"-- kline completeness: {r:.1f}%, now = ${close}" )
    
    def _r(x): # compare abs
        neg = x.iloc[-1]<0
        ocr = x.apply(abs).rolling(x.shape[0]).rank(pct=True).iloc[-1]
        ocr *=100
        return ocr, neg 
    if df.oc.iloc[-1]<0:     
        ocr, neg = _r( df[df.oc<0].dropna().oc )
    else:
        ocr, neg = _r( df[df.oc>0].dropna().oc )
    print(f'-- close minus open: $ {(close-open)}, { ((close-open)/open*10_000):.0f}bps')
    print(f'-- close open rank   ({"-" if neg else "+"}): {ocr:.1f}%')
    ocr, neg = _r( df.dropna().oc )
    print(f'-- close open rank (all): {ocr:.1f}%')
    hlr, neg = _r( df.dropna().hl )
    print(f'-- high low rank ({"-" if neg else "+"}): {hlr:.1f}%')

    cr, neg = _r( df.dropna().close )
    a,b= df.close.min(), df.close.max()
    print(f'-- close price rank: {cr:.1f}% (high=$ {b} ({((b-close)/close*100):.1f}%), low=$ {((a-close)/close*100):.1f}%)')
    
    print( f"-- current (UTC): {tnow}")

if __name__ == '__main__':
    main()