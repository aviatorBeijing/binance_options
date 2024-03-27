import os
import pandas as pd

from bbroker.settings import ex

def orders_status(ids=[]):
    if ids:
        pass 
    else: # all
        ods = ex.fetch_order_status()
        print(ods)

# tests
if __name__ == '__main__':
    orders_status()