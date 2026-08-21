import os, datetime, time
import pandas as pd
from tabulate import tabulate

from bbroker.settings import ex
from strategy.price_disparity import extract_specs
from strategy.delta_gamma import callprice, deltafunc, putprice
from butil.bsql import fetch_bidask 
from butil.butils import get_maturity, get_binance_spot, get_underlying, DEBUG

def orders_status() -> pd.DataFrame:
    ods = ex.eapiPrivateGetOpenOrders()
    df = pd.DataFrame.from_records(ods)
    if df.empty: 
        print('*** No outstanding orders.')
        return pd.DataFrame()

    target_cols = ['status', 'orderId', 'symbol', 'side', 'price', 'avgPrice', 'quantity', 'executedQty', 'updateTime', 'source', 'priceScale', 'quantityScale']
    
    # Safely select existing columns and fill missing target columns with None
    df = df.reindex(columns=target_cols)
    
    df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v / 1000, tz=datetime.timezone.utc))
    df = df.sort_values('updateTime', ascending=False)
    print('--[ orders ]\n', tabulate(df, headers="keys"))
    return df

def position_status() -> pd.DataFrame:
    ods = ex.eapiPrivateGetPosition()
    if not ods:
        print('-- no outstanding positions')
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(ods)
    
    target_cols = ['symbol', 'side', 'positionCost', 'ror', 'quantity', 'markValue', 'expiryDate']
    df = df.reindex(columns=target_cols)
    
    df['expiry'] = df.expiryDate.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v / 1000, tz=datetime.timezone.utc))
    df = df.sort_values(['symbol', 'expiryDate'], ascending=False)

    gain = df.markValue.astype(float).sum() - df.positionCost.astype(float).sum()
    print(f'-- positions gain (vs mark price): ${gain:.2f}')

    return df

# tests
def calc_(position_df):
    cs = list(position_df.symbol.values)
    position_df['spot'] = 0.; cnt = 0
    while True:
        try:
            display_df = position_df.copy()
            if cnt % 5 == 0:
                print('\n\n')
                orders_status()
                display_df['spot'] = display_df.symbol.apply(lambda c: get_binance_spot(get_underlying(c))[0])
            cnt += 1
            display_df['spec'] = display_df.symbol.apply(lambda s: extract_specs(s))
            display_df['K'] = display_df.spec.apply(lambda e: e[2])
            display_df['T'] = display_df.spec.apply(lambda e: e[1] / 365)
            display_df['spread'] = display_df.symbol.apply(fetch_bidask)
            display_df['bid'] = display_df.spread.apply(lambda e: float(e['bid']))
            display_df['ask'] = display_df.spread.apply(lambda e: float(e['ask']))
            display_df['impvol'] = display_df.spread.apply(lambda e: float(e['impvol']))

            display_df = display_df.drop(['spread', 'spec'], axis=1)
            
            display_df['gain'] = (display_df.bid * display_df.quantity.astype(float)) - display_df.positionCost.astype(float)
            display_df['gain%'] = (display_df.gain / display_df.positionCost.astype(float)) * 100
            
            display_df['bid'] = display_df['bid'].apply(lambda v: f"$ {v}")
            display_df['gain'] = display_df['gain'].apply(lambda v: f"$ {v:,.2f}")
            display_df['gain%'] = display_df['gain%'].apply(lambda s: f"{s:.2f}%")
        except Exception as e:
            print(str(e))
            print('*** waiting data:', cs)
        else:
            print('-- [ positions ]\n', tabulate(display_df, headers="keys"))
        time.sleep(5)

if __name__ == '__main__':
    print('*' * 30, ' Order Status', '*' * 30)
    orders_status()

    print('*' * 30, ' Existing Positions', '*' * 30)
    df = position_status()
    if df.empty:
        print('-- no outstanding positions')
    else:
        contracts = list(df.symbol.values)
        print('\n\n')

        from multiprocessing import Process
        from ws_bcontract import _main as ws_connector

        conn = Process(target=ws_connector, args=(",".join(contracts), "ticker",))
        calc = Process(target=calc_, args=(df,))
        conn.start()
        calc.start()
        
        conn.join()
        calc.join()
