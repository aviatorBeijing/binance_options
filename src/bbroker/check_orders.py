import os
import pandas as pd

from bbroker.settings import ex

def orders_status(ids=[]):
    if ids:
        pass 
    else: # all
        ods = ex.eapiPrivateGetOpenOrders()
        df = pd.DataFrame.from_records(ods)
        print(df)

# tests
if __name__ == '__main__':
    orders_status()