import os,click,datetime 
import pandas as pd 

fd = os.getenv('USER_HOME', '/Users/junma')

def daily_ohlcv_from_intraday(ric):
    fn = f'{fd}/tmp/{ric.replace("/","-").lower()}_1h.csv'
    ohlcv = pd.read_csv( fn )
    ohlcv.timestamp = ohlcv.timestamp.apply(pd.Timestamp)
    print(f'-- [ohlcv] latest: {ohlcv.iloc[-1].timestamp}')
    ohlcv.set_index('timestamp', inplace=True)
    for col in ['open','high','low','close','volume']: ohlcv[col] = ohlcv[col].apply(float)
    ohlcv = ohlcv.resample('1d').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    ohlcv.index = list(map(lambda s: str(s)[:10], ohlcv.index))
    return ohlcv

def pnl_from_trades( ric, tds, p0):
    print(f'-- [pnl (from trades)] reference price: ${p0}')
    tds = tds.copy()
    tds = tds[tds.symbol==ric.replace('/','').replace('-','').upper()]
    fee = tds.commission.astype(float).sum()
    gains = ( (p0 - tds.price) * tds.qty).sum()
    res = (tds.qty).sum()
    
    print(f'  -- res: {res} {ric}')
    print(f'  -- fee: $ {fee}')
    print(f'  -- gains: $ {gains:.6f} ($ {(gains-fee):.6f}, fee deducted)')

    return gains-fee, res

@click.command()
@click.option('--ric')
def main(ric):
    ohlcv = daily_ohlcv_from_intraday(ric)
    p0 = float(ohlcv.iloc[-1].close)

    odrs = f'{fd}/tmp/binance_perp_open_orders_{ric.replace("/","-").lower()}.csv'
    odrs = pd.read_csv( odrs )

    tds = f'{fd}/tmp/binance_perp_trades_{ric.replace("/","-").lower()}.csv'
    tds = pd.read_csv(tds)
    profits, res = pnl_from_trades( ric, tds, p0 )

    odrs['sign'] = odrs.side.apply(lambda e: -1 if e=='SELL' else 1)
    sz = (odrs.sign * odrs.origQty.astype(float) ).sum()
    print( f'-- open orders size: {sz} {ric}' )
    print( f'-- net position of trades & orders: {res + sz}' )
    print('-- [ open orders ]')
    print( odrs)

if __name__ == '__main__':
    main()
