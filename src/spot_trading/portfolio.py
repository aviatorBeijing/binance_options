import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from spot_trading.bs_meta import BianceSpot
from butil.butils import get_binance_spot

fd = os.getenv('USER_HOME',"/Users/junma")

def _aug_trades(tds,ric):
    tds = tds.copy()
    if 'neutral' not in tds:
        tds = tds[tds.symbol==ric.upper().replace('-','')]
        tds['sign'] = tds.side.apply(lambda s: 1 if s=='BUY' else -1)
        tds['qty'] = tds.sign * tds.qty.astype(float)
        tds['agg'] = tds.qty.cumsum()
        tds['$agg'] = -(tds.qty*tds.price.astype(float)).cumsum()
        tds['neutral'] = ''
        tds.loc[tds['agg']==0,'neutral'] = 'ok'
    return tds 

def _find_max_capital(tds:pd.DataFrame)->float:
    #tds = _aug_trades(tds,ric)
    capitals= []
    prev_idx = -1
    for i, idx in enumerate(tds[tds['neutral']=='ok'].index):
        if i ==0:
            max_capital = tds.loc[:idx]['$agg'].min()
        else:
            max_capital = tds.loc[prev_idx:idx]['$agg'].min()
        prev_idx = idx 
        capitals += [ max_capital]
    
    if prev_idx == -1: # Never happened
        max_capital = tds['$agg'].min()
    else:
        max_capital = tds.loc[prev_idx:]['$agg'].min() # cost is negative, so use min() here.
    capitals += [max_capital]
    mx = max( np.array(capitals)*-1. )
    return mx

def analyze_trades_cached(ric) -> pd.DataFrame:
    user_home = os.getenv('USER_HOME','')
    fn =user_home+f'/tmp/{ric.lower().replace("/","-")}_1h.csv'
    ohlcv = pd.read_csv( fn )
    ohlcv.timestamp = ohlcv.timestamp.apply(pd.Timestamp)
    ohlcv.set_index('timestamp', inplace=True)
    for col in ['open','high','low','close','volume']: ohlcv[col] = ohlcv[col].apply(float)
    ohlcv = ohlcv.resample('1d').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    ohlcv.index = list(map(lambda s: str(s)[:10], ohlcv.index))
    import talib
    ohlcv['atr'] = talib.ATR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    ohlcv['volume_rank'] = ohlcv.volume.rolling(14*2).rank(pct=True)*100

    ric = ric.lower().replace('/','-')
    fn = fd + f'/tmp/binance_trades_{ric}.csv'
    df = pd.read_csv(fn)
    df['datetime'] = df['datetime'].apply(pd.Timestamp)
    df.set_index('datetime',inplace=True)
    print(f'-- [trades from cached: {fn}]')
    print(df)

    df = _aug_trades( df, ric )
    max_capital = _find_max_capital(df)
    max_equity_amt = -(df.qty.cumsum().min())
    p = df[df['neutral']=='ok']['$agg']/max_capital*100

    max_eq = max_equity_amt * df.iloc[0].price
    capital_usage = max_capital + max_eq
    print(f'-- capital usage: ${capital_usage:,.2f}')
    print(f'  -- max cash: \t\t${max_capital:.2f}')
    print(f'  -- max equity({ric.upper()}): {max_equity_amt} (${(max_eq):,.2f})')

    sym = ric.upper().split('/')[0].split('-')[0]
    fig, ((ax1,ax7), (ax01,ax02), (ax5,ax3) )= plt.subplots(3,2,figsize=(27,15))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    ax2 = ax1.twinx()
    ax8 = ax02.twinx()
    
    df['cash'] = df['$agg']
    df['asset'] = df.qty.cumsum()
    fee_cumsum = df['fee'].astype(float).cumsum()
    df['portfolio'] = df['cash'] + df['asset']*df.price -fee_cumsum
    port = df[['price','portfolio']].resample('1d').agg('last')# * 2/3
    port.index = list(map(lambda d: str(d)[:10], port.index))
    port = pd.concat([port, ohlcv[['close']]], axis=1, ignore_index=False).dropna()
    
    ax1.set_ylabel('profit %', color = 'blue') 
    ax2.set_ylabel('equity $', color = 'orange') 
    ax2.set_title(f'Net return v.s. spot ({sym})\n'
                +f'max cash: ${max_capital:.2f}\n'
                +f'max equity: ${abs(max_eq):.2f} (#{abs(max_equity_amt)})\n'
                +f'fee \${fee_cumsum.iloc[-1]:.2f}')
    (port.portfolio/capital_usage*100).plot(ax=ax1,color='blue',linewidth=5)
    (port.close).plot(ax=ax2,color='orange')
    ax2.grid()

    ax77 = ax7.twinx()
    daily_vol = pd.concat([port[['portfolio']], ohlcv[['atr']]],axis=1,ignore_index=False).dropna()
    (daily_vol.portfolio/capital_usage*100).plot(ax=ax7,color='blue',linewidth=3)
    daily_vol.atr.plot(ax=ax77,color='red')
    ax7.set_ylabel(f'profit %',color='blue') 
    ax7.set_title('Return v.s. ATR')
    ax77.set_ylabel(f'ATR',color='red') 

    ax011 = ax01.twinx()
    (port.portfolio/capital_usage*100).plot(ax=ax01,color='blue',linewidth=3)
    _vrank = pd.concat([ohlcv[['volume_rank']], port.portfolio], axis=1,ignore_index=False).dropna()
    _vrank['volume_rank'].plot(ax=ax011,color='red',linestyle='--') 
    ax01.set_ylabel(f'portfolio %') 
    ax01.set_title('Return v.s. market volume')
    ax011.set_ylabel(f'Mkt. volume ranking %',color='red') 
    
    _x = df.qty.resample('1d').agg('sum')
    _xx = _x.copy() 
    if _xx.shape[0]>30: # Bar plotting will not be consolidate automatically, hence can be messy. 
        _xx = _xx[-30:]
    _xx.index = list(map(lambda d: str(d)[:10], _xx.index))
    #_x.index = range(0, _x.shape[0])
    _xx.plot.bar(ax=ax3)
    ax3.set_title(f'Daily positions changes {ric.upper().split("-")[0]}')
    #for tic in ax3.get_xticklabels(): tic.set_rotation(35)

    df['zeros'] = 0.
    _x = df[['qty']].cumsum()
    ax5.step( _x.index, _x.qty.values,color = 'blue' )
    df.zeros.plot(ax=ax5,color='grey',linestyle='--')
    ax5.set_ylabel(f'{ric.upper().split("-")[0]} #', color = 'blue') 
    ax5.set_title(f'Agg. positions ({ric.upper().split("-")[0]})')
    ax5.grid()

    daily_vol = df[['qty']].apply(abs).resample('1d').agg('sum')
    daily_vol.index = list(map(lambda s: str(s)[:10], daily_vol.index))
    daily_vol.plot(ax=ax02)
    _vrank = pd.concat([ohlcv[['volume_rank']], daily_vol], axis=1,ignore_index=False).dropna()
    _vrank['volume_rank'].plot(ax=ax8,color='red',linestyle='--') 
    #for tic in ax7.get_xticklabels(): tic.set_rotation(35)
    ax02.set_ylabel(f'# {ric.upper().split("-")[0]}') 
    ax02.set_title('Trading qty v.s. market volume')
    ax8.set_ylabel(f'Mkt. volume ranking %',color='red') 

    fn =f"{fd}/tmp/binance_portfolio_{ric.lower().replace('/','-')}.png"
    plt.savefig(fn)
    print('-- saved portfolio:', fn)
    return df

def read_cached_trades(ric):
    ric = ric.lower().replace('/','-')
    fn = fd + f'/tmp/binance_trades_{ric}.csv'
    #fn = fd + f'/tmp/binance_trades.csv'
    if os.path.exists(fn):
        df = pd.read_csv( fn, index_col=False)
        df['index'] = df['id'];df.set_index('index',inplace=True)
        return df 
    return pd.DataFrame()

def analyze_trades(ric, tds, days, save=True):
    old_tds = read_cached_trades(ric)
    tds = tds.copy()
    if not old_tds.empty:
        old_tds['id'] = old_tds['id'].apply(int)
        if not tds.empty:
            tds['id'] = tds['id'].apply(int)
            tds['index'] = tds['id'];tds.set_index('index',inplace=True)
            tds = pd.concat([old_tds,tds], axis=0, ignore_index=False)
        else:
            tds = old_tds
    tds = tds.sort_values('id').drop_duplicates(subset=['id'],keep="first",ignore_index=False)    
    if save:
        ric = ric.lower().replace('/','-')
        fn = fd + f'/tmp/binance_trades_{ric}.csv'
        #fn = fd + f'/tmp/binance_trades.csv'
        for col in 'qty,price,commission'.split(','):
            tds[col] = tds[col].apply(float)
        tds['datetime'] = tds['datetime'].apply(str)

        pceMap = {}
        syms = list(set(tds.commissionAsset.values))
        for s in syms:
            if s =='USDT':
                pceMap[s] = 1.
            else:
                feeric = f'{s}/USDT'
                bid,ask = get_binance_spot( feeric )
                pceMap[s] = bid
        tds['commAssetPrice'] = tds.commissionAsset.apply(lambda s: pceMap[s])
        tds['fee'] = tds.commission.astype(float)*tds.commAssetPrice

        tds.to_csv(fn,index=False)
        print('-- saved:', fn)
    print(f"-- Total: {tds.shape[0]}, start: {tds.iloc[0]['datetime']}")
   
    tds = tds[tds.symbol==ric.upper().replace('-','')]

    tds['sign'] = tds.side.apply(lambda s: 1 if s=='BUY' else -1)
    tds['qty'] = tds.sign * tds.qty.astype(float)
    tds['agg'] = tds.qty.cumsum()
    tds['$agg'] = -(tds.qty*tds.price.astype(float)).cumsum()
    tds['neutral'] = ''
    tds.loc[tds['agg']==0,'neutral'] = 'ok'
    print('-- [trades]')
    if tds.shape[0]>10:
        print( tabulate(tds.head(3),headers="keys") )
    print( tabulate(tds.tail(10),headers="keys") )
    return tds

def calc_avg_holding_price( tds = pd.DataFrame()) -> tuple:
    assert 'neutral' in tds and 'commAssetPrice' in tds, f"trades dataframe should have two special columns"
    neu_indices = list( tds[tds.neutral=='ok'].index )
    #assert len(neu_indices)>0, f"no neutral location found in trades history"
    if len(neu_indices)>0:
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

def portfolio_check(ric,days=3):
    """
    @param days (int): how long to look back for trades
    """
    from bbroker.settings import spot_ex
    mkt = BianceSpot(ric.replace('-','/'), spot_ex=spot_ex)
    
    print(f'*** fetching trades data in {days} days. Be aware, max count returned by Binance API is 500 trades. To ensure getting all trades, the day count might need to be adjusted to set the start time of fetching.')
    tds = mkt.check_trades(hours=days*24)
    tds = analyze_trades( ric, tds, days)
    
    if 'commAssetPrice' not in tds:
        pceMap = {}
        syms = list(set(tds.commissionAsset.values))
        for s in syms:
            if s =='USDT':
                pceMap[s] = 1.
            else:
                feeric = f'{s}/USDT'
                bid,ask = get_binance_spot( feeric )
                pceMap[s] = bid
        tds['commAssetPrice'] = tds.commissionAsset.apply(lambda s: pceMap[s])
    fee = (tds.commission.astype(float)*tds.commAssetPrice).sum()
   
    sym = ric.upper().split("-")[0].split("/")[0]
    n_doge = 0.
    for feeasset in list(set(tds.commissionAsset.values)):
        feex = tds[tds.commissionAsset==feeasset].commission.astype(float).sum()
        print(f'  -- #fee in {feeasset}: {feex}')
        if feeasset == sym: #'DOGE':
            n_doge = feex

    net_doge_deficit = tds.iloc[-1]['agg'] - n_doge
    print(f'  -- net {sym} deficite: {net_doge_deficit:.0f}')
    
    pce,_ = get_binance_spot( ric.replace('-','/') ) # price now
    port_value = tds.iloc[-1]['agg'] * pce  + tds.iloc[-1]['$agg'] - fee # position value + cash changes - fee
    holding_cost, holding_size = calc_avg_holding_price( _aug_trades(tds,ric) )

    print(f'-- fee: ${fee:4f} {((fee/(fee+port_value))*100):.1f}%')
    print(f'-- holding: {holding_size} shares, average cost: $ {holding_cost:.4f}')
    print(f'-- gain (after liquidating and fee deduction @ ${pce}): $ {port_value:,.4f}')
    
    # orders
    openDf = mkt.check_open_orders()

    res = openDf.shape[0]
    fn = fd + f'/tmp/binance_fee_gain_{ric.lower().replace("/","-")}.dat'
    with open(fn,'w') as fp:
        fp.writelines([f'ric:{ric}\n', f'fee:${fee:4f}\n',f'gain:${port_value:,.4f}\n',f'price:${pce}\n',f'holding:{holding_size}\n',f'orders:{res}'])

def assets():
    fn = os.getenv('USER_HOME','') + "/tmp/bal.csv"
    if os.path.exists( fn ):
        df = pd.read_csv( fn )
        df['ref'] = df['value'] / df['ttl']
        #df = df[df.asset != 'BTC']

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8))
        dim = df.shape[0]
        w = 0.75
        dimw = w / 2
        x = np.arange( df.shape[0] )
        ax1.bar(x, df['ttl'], dimw, bottom=0.001)
        ax1.bar(x + dimw, df['free'], dimw, bottom=0.001)
        #ax1.bar(x + 2*dimw, df['locked'], dimw, bottom=0.001)
        
        ax1.set_xticks(x + dimw / 2, labels=map(str, x))
        ax1.set_yscale('log')
        ax1.set_ylabel('count')
        ax1.set_xlabel('')
        ax1.set_xticks( x, df.asset, rotation=15 ) 

        wedges, texts, autotexts = ax2.pie(
            df['value'],labels=df['asset'], autopct='%1.1f%%')
        ax1.legend(
            wedges, 
            df['asset'].apply(lambda s: s[:3]) + df['value'].apply(lambda v: f"${v:.2f}"),
            title="Assets",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
            )

        fn = os.getenv("USER_HOME",'')+'/tmp/bal.png'
        plt.savefig(fn)
        print('-- saved:', fn)
    else:
        print(f'*** {fn} doesn\'s exit.')

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--days',default=3)
#@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
@click.option('--check_cached', is_flag=True, default=False)
@click.option('--spot', default=0.155, help="the current spot price (mainly used for offline purpose)")
def main(ric,days,check_cached,spot): 
    if check_cached:
        _ = analyze_trades_cached(ric)
        _ = BianceSpot.analyze_open_orders_cached(spot,ric)
        _ = assets()
    else:   
        portfolio_check(ric,days=days)

if __name__ == '__main__':
    main()
