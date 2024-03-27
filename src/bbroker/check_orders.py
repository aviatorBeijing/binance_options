import os
import pandas as pd
from tabulate import tabulate

from bbroker.settings import ex

def orders_status(ids=[]):
    if ids:
        pass 
    else: # all
        ods = ex.eapiPrivateGetOpenOrders()
        df = pd.DataFrame.from_records(ods)
        df = df[['orderId','symbol','side','price','avgPrice','quantity','executedQty','status','updateTime','source','clientOrderId','priceScale','quantityScale']]
        print(tabulate(df,headers="keys"))

# tests
if __name__ == '__main__':
    orders_status()