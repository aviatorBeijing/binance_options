import os,datetimme
import pandas as pd 

from butil.butils import get_binance_index

def cunit(contract):
    sz = 0
    if contract.startswith( 'BTC-' ):
        sz = 1
    elif contract.startswith( 'XRP-' ):
        sz = 100
    elif contract.startswith( 'DOGE-' ):
        sz = 1_000
    else: raise Exception(f"{contract} is not supported.")
    return sz 

def calc_fee(trade_price, trade_vol, contract, is_taker=True):
    '''
    Ref:
    https://www.binance.com/en/support/faq/binance-options-trading-fees-5326e5de61c34fed98abe28d2f175a23

    Transaction Fee = 
        Minimum (
            Transaction Fee Rate * Index Price * Contract Unit, 
            10% * Option Traded Price
        ) * Option Traded Size
    '''
    fee_rate = 5/10000
    if not is_taker:
        fee_rate = 2/10000
    index_price = get_binance_index( contract )
    contract_unit = cunit( contract )

    return min(
        fee_rate * index_price * contract_unit,
        trade_price * 0.1
    ) * trade_vol
