import os,datetime
import pandas as pd 

from butil.butils import get_binance_index

def cunit(contract):
    # https://www.binance.com/en/support/faq/binance-options-contract-specifications-cdee5d43b70d4d2386980d41786a8533
    sz = 0
    if contract.startswith( 'BTC-' ) or \
        contract.startswith( 'ETH-' ) or \
            contract.startswith( 'BNB-' ):
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
    fee_rate = 3/10000 # FIXME: for DOGE- contract, it's 3/10000 (see the BN trading UI)
    if not is_taker:
        fee_rate = 2/10000
    index_price = get_binance_index( contract )
    contract_unit = cunit( contract )

    return min(
        fee_rate * index_price * contract_unit,
        trade_price * 0.1
    ) * trade_vol
