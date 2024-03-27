import os,datetime
import pandas as pd
from tabulate import tabulate

from bbroker.settings import ex

def orders_status(ids=[]):
    #tnow = datetime.datetime.utcnow().timestamp()*1000;tnow=int(tnow)
    if ids:
        pass 
    else: # all
        ods = ex.eapiPrivateGetOpenOrders()
        df = pd.DataFrame.from_records(ods)
        #df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df[['status','orderId','symbol','side','price','avgPrice','quantity','executedQty','updateTime','source','priceScale','quantityScale']]
        df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
        df = df.sort_values('updateTime', ascending=False)
        print(tabulate(df,headers="keys"))

def position_status():
    ods = ex.eapiPrivateGetPosition()
    """
    'entryPrice': '55.5', 'symbol': 'ETH-240329-3550-P', 'side': 'LONG', 
    'quantity': '0.1', 'reducibleQty': '0', 'markValue': '4.64', 'ror': '-0.1639', 
    'unrealizedPNL': '-0.91', 'markPrice': '46.4', 'strikePrice': '3550.00000000', 
    'positionCost': '5.55', 'expiryDate': '1711699200000', 'priceScale': '1', 
    'quantityScale': '2', 'optionSide': 'PUT', 'quoteAsset': 'USDT', 'time': '1711500354804'
    """
    df = pd.DataFrame.from_records(ods)
    df = df[['status','orderId','symbol','side','positionCost',
                'entryPrice','markPrice','quantity','markValue','updateTime','expiryDate']]
    df['datetime'] = df.time.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
    df = df.sort_values('time', ascending=False)
    print(tabulate(df,headers="keys"))

# tests
if __name__ == '__main__':
    orders_status()
    position_status()