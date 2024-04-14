import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from spot_trading.bs_meta import BianceSpot
from butil.butils import binance_spot

fd = os.getenv('USER_HOME',"/Users/junma")

def analyze_trades_cached(days=72) -> pd.DataFrame:
    fn = fd + f'/tmp/binance_trades_in_{days}.csv'
    df = pd.read_csv(fn)
    print(f'-- [trades from cached: {fn}]')
    print(df)
    return df

def analyze_trades(ric, tds, days, save=True):
    tds = tds[tds.symbol==ric.replace('-','')]
    tds['sign'] = tds.side.apply(lambda s: 1 if s=='BUY' else -1)
    tds['qty'] = tds.sign * tds.qty.astype(float)
    tds['agg'] = tds.qty.cumsum()
    tds['$agg'] = -(tds.qty*tds.price.astype(float)).cumsum()
    tds['neutral'] = ''
    tds.loc[tds['agg']==0,'neutral'] = 'ok'
    print('-- [trades]')
    print( tds )
    if save:
        fn = fd + f'/tmp/binance_trades_in_{days}.csv'
        tds.to_csv(fn,index=0)
        print('-- saved:', fn)

def portfolio_check(ric,days=72):
    """
    @param days (int): how long to look back for trades
    """
    from bbroker.settings import spot_ex
    mkt = BianceSpot(ric.replace('-','/'), spot_ex=spot_ex)
    
    tds = mkt.check_trades(hours=days*24)
    analyze_trades( ric, tds, days)
    
    pceMap = {}
    syms = list(set(tds.commissionAsset.values))
    for s in syms:
        if s =='USDT':
            pceMap[s] = 1.
        else:
            feeric = f'{s}/USDT'
            bid,ask = binance_spot( feeric )
            pceMap[s] = bid
    tds['commAssetPrice'] = tds.commissionAsset.apply(lambda s: pceMap[s])
    fee = (tds.commission.astype(float)*tds.commAssetPrice).sum()
    print(f'-- fee: ${fee:4f}')

    pce,_ = binance_spot( ric.replace('-','/') )
    port_value = tds.iloc[-1]['agg'] * pce  + tds.iloc[-1]['$agg'] - fee 
    print(f'-- gain (after liquidating): $ {port_value:,.4f}')
    fn = fd + f'/tmp/binance_fee_gain.dat'
    with open(fn,'w') as fp:
        fp.writelines([f'fee:${fee:4f}\n',f'gain:${port_value:,.4f}'])
    
    # orders
    openDf = mkt.check_open_orders()

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--days',default=72)
#@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
@click.option('--check_cached', is_flag=True, default=False)
def main(ric,days,check_cached): 
    if check_cached:
        _ = analyze_trades_cached(72)
        _ = BianceSpot.analyze_open_orders_cached()
    else:   
        portfolio_check(ric,days=days)

if __name__ == '__main__':
    main()
