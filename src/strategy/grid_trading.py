import pandas as pd 
import click,os
import numpy as np 
from tabulate import tabulate

from butil.butils import binance_kline

# Toy strategy

def paper_trading(df, max_pos):
    df['avg'] = (df.open+df.close+df.high+df.low)/4
    
    # trading price
    df['buy'] = df.avg
    df['sell'] = df.avg
    
    # significance
    df['insignificant'] = ((df.close-df.open).apply(abs).rolling(60).rank(pct=True)<0.75).shift(1)
    #df['insignificant'] =( ( (df.close-df.open)/(df.high-df.low)).apply(abs) < 0.6 ).shift(1)

    df['prev_up'] = (df['close'] > df['open']).shift(1)
    df['prev_down'] = (df['close'] < df['open']).shift(1)
    
    # Pseudo-trading
    df = df.dropna()
    df.loc[df.insignificant, 'sell'] = 0
    df.loc[df.insignificant, 'buy'] = 0
    df.loc[df.prev_up, 'sell'] = 0 # buy
    df.loc[df.prev_down, 'buy'] = 0 # sell
    #print(df)

    pos = 0;portfolio = []; profits = [];vol=0.0
    buys = 0;sells=0
    for i,row in df.iterrows():
        profit = 0
        if row.buy > 0 and pos < max_pos:
            portfolio+= [ row.buy ]
            vol += row.buy
            pos +=1
            buys+=1
        elif row.sell > 0 and portfolio:
            sold = portfolio.pop()
            pos -=1
            profit = sold - row.sell
            vol += row.sell
            sells+=1
        profits += [(i, profit)]
        #print( pos, profit )
    
    res = 0.
    if portfolio:
        res = len(portfolio)*df.iloc[-1].close - np.sum(portfolio) 

    fee = 1e-3
    fee = vol*fee + len(portfolio)*df.iloc[-1].close*fee
    ttl = np.sum(list( map(lambda e: e[1],profits)) )
    net_ttl =  ttl + res - fee
    
    return buys,sells,res,fee,ttl,net_ttl

@click.command()
@click.option('--ric', default="BTC/USDT")
@click.option('--span', default='1h')
@click.option('--test', default=False, is_flag=True)
@click.option('--max_pos', default=50)
@click.option('--nominal', default=1.0, help="scale up or down the trading size for small valued assets")
def main(ric,span,test,max_pos,nominal):
    print('-- max pos:',max_pos, ' (i.e., sizing)')
    recs = []
    for span in ['1h','8h','1d']:
        fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
        if not test:
            df = binance_kline(ric, span=span)
            df.to_csv(fn)
            print('-- saved:', fn)
        else:
            #print('-- reading:',fn)
            df = pd.read_csv( fn )
        
        # subsets
        for n in [0,100,300,400]:
            #print('--', f'span={span},', df.iloc[n].timestamp, '~', df.iloc[n+550].timestamp)
            idf = df.copy()[n:n+550]
            buys,sells,res,fee,ttl,net_ttl = paper_trading(idf,max_pos)
            #print( '--', f'span={span},', df.iloc[n].timestamp, '~', df.iloc[n+550].timestamp, f'\t{buys}:{sells} \t res= ${res:,.0f} \tfee= ${fee:,.0f} \tcash= ${ttl:,.0f} \tnet= ${net_ttl:,.0f} \t{(fee/(fee+net_ttl)*100):.1f}%')
            recs += [(span,df.iloc[n].timestamp,df.iloc[n+550].timestamp,buys,sells,res,fee,ttl,net_ttl)]
        buys,sells,res,fee,ttl,net_ttl =  paper_trading(df,max_pos)
        recs += [(span,df.iloc[0].timestamp,df.iloc[-1].timestamp,buys,sells,res,fee,ttl,net_ttl)]
        #print( '--', f'span={span},', df.iloc[0].timestamp, '~', df.iloc[-1].timestamp, f'\t{buys}:{sells} \t res= ${res:,.0f} \tfee= ${fee:,.0f} \tcash= ${ttl:,.0f} \tnet= ${net_ttl:,.0f} \t{(fee/(fee+net_ttl)*100):.1f}%')

    df = pd.DataFrame.from_records( recs )
    df.columns='span,t1,t2,buys,sells,res,fee,ttl,net_ttl'.split(',')
    df['fee%'] = df.fee/(df.fee+df.net_ttl)*100
    for col in 'res,fee,ttl,net_ttl'.split(','):
        df[col] = df[col].apply(lambda v: v*nominal)
    for col in 'res,fee,ttl,net_ttl,fee%'.split(','):
        df[col] = df[col].apply(lambda s: f"{s:,.0f}" if col != 'fee%' else f"{s:.1f}%")
    print(tabulate(df, headers="keys"))
if __name__ == '__main__':
    main()