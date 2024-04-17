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
    df['datetime'] = df['datetime'].apply(pd.Timestamp)
    df.set_index('datetime',inplace=True)
    print(f'-- [trades from cached: {fn}]')
    print(df)

    import matplotlib.pyplot as plt 
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.grid()

    ax1.set_ylabel('$agg', color = 'blue') 
    ax2.set_ylabel('#agg', color = 'red') 
    ax1.plot(df['$agg'], color='blue')
    ax2.plot(df['agg'], color='red')
    fn =f"{fd}/tmp/binance_portfolio.png"
    plt.savefig(fn)
    print('-- saved:', fn)
    return df

def analyze_trades(ric, tds, days, save=True):
    tds = tds.copy()
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
    return tds

def calc_avg_holding_price( tds = pd.DataFrame()) -> tuple:
    assert 'neutral' in tds and 'commAssetPrice' in tds, f"trades dataframe should have two special columns"
    neu_indices = list( tds[tds.neutral=='ok'].index )
    assert len(neu_indices)>0, f"no neutral location found in trades history"
    tds = tds.iloc[ neu_indices[-1]+1: ]
    if tds.empty:
        return 0.,0.
    res_size = tds.qty.sum()
    if res_size==0:
        return 0.,0.
    fee = (tds.commission.astype(float)*tds.commAssetPrice).sum()
    cost = (tds.qty.astype(float)*tds.price.astype(float)).sum() + fee
    res_price = cost/res_size
    return res_price, res_size

def portfolio_check(ric,days=72):
    """
    @param days (int): how long to look back for trades
    """
    from bbroker.settings import spot_ex
    mkt = BianceSpot(ric.replace('-','/'), spot_ex=spot_ex)
    
    tds = mkt.check_trades(hours=days*24*2)
    tds = analyze_trades( ric, tds, days)
    
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

    holding_cost, holding_size = calc_avg_holding_price( tds )
    print(f'-- holding: {holding_size} shares, average cost: $ {holding_cost:.4f}')

    pce,_ = binance_spot( ric.replace('-','/') )
    port_value = tds.iloc[-1]['agg'] * pce  + tds.iloc[-1]['$agg'] - fee 
    print(f'-- gain (after liquidating and fee deduction): $ {port_value:,.4f}')
    
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
@click.option('--spot', default=0.155, help="the current spot price (mainly used for offline purpose)")
def main(ric,days,check_cached,spot): 
    if check_cached:
        _ = analyze_trades_cached(72)
        _ = BianceSpot.analyze_open_orders_cached(p0=spot)
    else:   
        portfolio_check(ric,days=days)

if __name__ == '__main__':
    main()
