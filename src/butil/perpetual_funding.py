import datetime,os,click
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt 

user_home = os.getenv('USER_HOME','/Users/junma')
cached_history = user_home+"/tmp/{}_perpetual.csv"

"""
Results:
                 close       vol       rtn   funding  funding_prev  funding_next
close         1.000000  0.373806  0.080187  0.645449      0.615975      0.635368
vol           0.373806  1.000000  0.062743  0.484463      0.359417      0.545908
rtn           0.080187  0.062743  1.000000  0.038636      0.011763      0.091579
funding       0.645449  0.484463  0.038636  1.000000      0.797124      0.797124
funding_prev  0.615975  0.359417  0.011763  0.797124      1.000000      0.698176
funding_next  0.635368  0.545908  0.091579  0.797124      0.698176      1.000000

From this correlation matrix, it seems "funding rate" is HIGHLY corrleated to the prices,
instead of returns. So, it's the highly IRRATIONAL sentiment that drives funding rates to go high.
"""
def get_spot_history(sym)->pd.DataFrame:
    print(f'-- [{sym} Spot]')
    spot = pd.read_csv(user_home+f"/tmp/{sym.lower()}_spot.csv",index_col=False)
    df = spot[['Timestamp','Open','High','Low','Close','Volume']]
    df.columns = list(map(lambda s: s.lower(),df.columns))
    df.loc[:,'timestamp'] = df.timestamp.apply(datetime.datetime.fromtimestamp)\
                                    .apply(pd.Timestamp)
    df.loc[:,'vol'] = df.close.fillna(0).rolling(3*7).agg(np.std)
    df = df[['timestamp','close','vol']]
    df.set_index(['timestamp'],inplace=True)
    df.loc[:,'close'] = df.close.resample('1h').agg('last')#.rolling(3*7).agg(np.mean)
    df.loc[:,'rtn'] = df.close.fillna(0).pct_change()
    df.dropna(inplace=True)
    return df

def get_funding_history(sym)->pd.DataFrame:
    print(f'-- [{sym} Perp]')
    hist = pd.read_csv( cached_history.format(sym.lower() ) )
    df = hist[['Time','Funding Rate']]
    df.columns = ['datetime', 'funding']
    df.loc[:,'funding'] = df.funding.apply(lambda s: float(s.replace('%','')))
    df = df.sort_values(['datetime'], ascending=True)
    df.loc[:,'datetime'] = df['datetime']\
                            .apply(lambda e: datetime.datetime.strptime(e, '%Y-%m-%d %H:%M:%S'))\
                            .apply(pd.Timestamp)
    df.set_index('datetime', inplace=True,drop=True)
    return df 

@click.command()
@click.option('--symbol', default='BTCUSDT', help='ex. BTCUSDT (for BTC perpetual)')
def main(symbol):
    df = get_funding_history(symbol)
    spot = get_spot_history(symbol )
    df = pd.concat([spot,df],axis=1,ignore_index=False).ffill()
    df.dropna(inplace=True)
    #print( tabulate(df,headers="keys"))
    print(df)

    df['funding_prev'] = df.funding.shift(8)
    df['funding_next'] = df.funding.shift(-8)
    
    print( df.corr() )
    #df[['rtn','funding']].plot();plt.show()

if __name__ == '__main__':
    main()
