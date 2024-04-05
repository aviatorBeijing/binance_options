import os,datetime,time
import pandas as pd
from tabulate import tabulate

from bbroker.settings import ex
from strategy.price_disparity import extract_specs
from strategy.delta_gamma import callprice, deltafunc,putprice

def orders_status()->pd.DataFrame:
    #tnow = datetime.datetime.utcnow().timestamp()*1000;tnow=int(tnow)
    ods = ex.eapiPrivateGetOpenOrders()
    df = pd.DataFrame.from_records(ods)
    #df['dt'] = (tnow - df.updateTime.apply(int))/1000
    df = df[['status','orderId','symbol','side','price','avgPrice','quantity','executedQty','updateTime','source','priceScale','quantityScale']]
    df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
    df = df.sort_values('updateTime', ascending=False)
    print(tabulate(df,headers="keys"))
    return df

def position_status()->pd.DataFrame:
    ods = ex.eapiPrivateGetPosition()
    """
    'entryPrice': '55.5', 'symbol': 'ETH-240329-3550-P', 'side': 'LONG', 
    'quantity': '0.1', 'reducibleQty': '0', 'markValue': '4.64', 'ror': '-0.1639', 
    'unrealizedPNL': '-0.91', 'markPrice': '46.4', 'strikePrice': '3550.00000000', 
    'positionCost': '5.55', 'expiryDate': '1711699200000', 'priceScale': '1', 
    'quantityScale': '2', 'optionSide': 'PUT', 'quoteAsset': 'USDT', 'time': '1711500354804'
    """
    if not ods:
        print('-- no outstanding positions')
        return pd.DataFrame()
    df = pd.DataFrame.from_records(ods)
    df = df[['symbol','side','positionCost','ror',
            'quantity','markValue','expiryDate']]
    df['expiry'] = df.expiryDate.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
    df = df.sort_values(['symbol','expiryDate'], ascending=False)
    print(tabulate(df,headers="keys"))

    gain = df.markValue.astype(float).sum()-df.positionCost.astype(float).sum()
    print(f'-- gain (vs mark price): ${gain:.2f}' )

    return df

# tests
def calc_(position_df):
    from butil.bsql import fetch_bidask
    from butil.butils import get_maturity,get_binance_spot,get_underlying, DEBUG


    cs = list(position_df.symbol.values)
    position_df['spot'] = 0.;cnt = 0
    while True:
        try:
            if cnt%5 == 0:
                print('\n\n')
                orders_status()
                position_df['spot'] = position_df.symbol.apply(lambda c: get_binance_spot( get_underlying(c) )[0])
            cnt +=1
            position_df['spec'] = position_df.symbol.apply(lambda s: extract_specs( s ) )
            position_df['K'] = position_df.spec.apply(lambda e: e[2])
            position_df['T'] = position_df.spec.apply(lambda e: e[1]/365)
            position_df['spread'] = position_df.symbol.apply(fetch_bidask)
            position_df['bid'] = position_df.spread.apply(lambda e: float(e['bid']))
            position_df['ask'] = position_df.spread.apply(lambda e: float(e['ask']))
            position_df['impvol'] = position_df.spread.apply(lambda e: float(e['impvol']))
            #putprice(spot_price, K, T/365, sigma, r )


            position_df = position_df.drop(['spread','spec',],axis=1)
            
            position_df['gain'] = (position_df.bid *position_df.quantity.astype(float)) - position_df.positionCost.astype(float)
            position_df['gain%'] = (position_df.gain / position_df.positionCost.astype(float))*100
            
            position_df['bid'] = position_df['bid'].apply(lambda v: f"$ {v}")
            position_df['gain'] = position_df['gain'].apply(lambda v: f"$ {v:,.2f}")
            position_df['gain%'] = position_df['gain%'].apply(lambda s: f"{s:.2f}%")
        except Exception as e:
            print(str(e))
            print('*** waiting data:', cs )
        else:
            print( tabulate(position_df,headers="keys") )
        time.sleep(5)

if __name__ == '__main__':
    print('*'*30, ' Order Status', '*'*30)
    orders_status()

    print('*'*30, ' Existing Positions', '*'*30)
    df = position_status()
    if df.empty:
        print('-- no outstanding positions')
    else:
        contracts = list(df.symbol.values)
        print('\n\n')

        from multiprocessing import Process
        from ws_bcontract import _main as ws_connector

        conn = Process( target=ws_connector, args=(",".join(contracts), "ticker",) )
        calc = Process( target=calc_, args=(df,) )
        conn.start()
        calc.start()
        
        conn.join()
        calc.join()
