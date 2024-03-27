import os,datetime
import pandas as pd
from tabulate import tabulate

from bbroker.settings import ex

def orders_status(ids=[]):
    tnow = datetime.datetime.utcnow().timestamp()*1000;tnow=int(tnow)
    if ids:
        pass 
    else: # all
        ods = ex.eapiPrivateGetOpenOrders()
        df = pd.DataFrame.from_records(ods)
        df = df[['status','orderId','symbol','side','price','avgPrice','quantity','executedQty','updateTime','source','clientOrderId','priceScale','quantityScale']]
        df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df.sort_values('updateTime', ascending=False)
        print(tabulate(df,headers="keys"))

# tests
if __name__ == '__main__':
    orders_status()