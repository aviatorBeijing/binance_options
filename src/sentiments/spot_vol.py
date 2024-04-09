import requests,os,datetime,click
import pandas as pd 
import socks,socket

YF="https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&period1={}&period2={}" #1709525226

def using_proxy():
    if os.getenv("YAHOO_LOCAL", None):
        print('-- using proxy')
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 50000)
        socket.socket = socks.socksocket

def ydata(ric,startts,endts):
    using_proxy()
    resp = requests.get(
        YF.format(ric.upper(), int(startts), int(endts)),
        headers={
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'
                }
    )
    return resp.json()

def _main(ric,check='return'): # check='return' | 'gamma'  (gamma is the derivative of return)
    fn = f"{ric.lower()}.csv"
    if True: #not os.path.exists(fn):
        """endts = (datetime.datetime.utcnow()+datetime.timedelta(hours=8)).timestamp()
        startts = endts - 10*365*24*3600
        resp = ydata( ric, startts,endts)
        timestamps = resp['chart']['result'][0]['timestamp']
        data = resp['chart']['result'][0]['indicators']['quote'][0]
        
        df = pd.DataFrame.from_dict({"date": timestamps, 
                                    "open": data['open'],
                                    "high": data['high'],
                                    "low": data['low'],
                                    "close": data['close'] })
        """
        from butil.butils import binance_kline
        df = binance_kline(f"{ric.upper()}/USDT")
        startts = df.timestamp.iloc[0]
        endts = df.timestamp.iloc[-1]
        #df.to_csv(fn, index=False)
    else:
        df = pd.read_csv(fn)
        startts = df.timestamp.iloc[0]
        endts = df.timestamp.iloc[-1]
    
    df.timestamp = df.timestamp.apply(pd.Timestamp)
    #df.set_index('date',drop=True,inplace=True)
    df['week_days']= df.timestamp.apply(lambda e: e.dayofweek)
    
    df.dropna(inplace=True)
    df['rtn'] = df.close.pct_change()
    df['gamma'] = df.rtn.pct_change()
    df['gamma_rnk'] = df.rtn.pct_change().rolling(30).rank(pct=True)
    last_row = df.tail(1)
    last_rtn = df.rtn.iloc[-1]
    last_rtn_rk = df.rtn.rolling(df.shape[0]).rank(pct=True).iloc[-1]*100
    df.dropna(inplace=True)
    last_gamma = df.gamma.iloc[-1]
    last_gamma_rnk = df.gamma_rnk.iloc[-1]
    print( last_row )
    
    col = 'rtn' if check == 'return' else 'gamma' if check == 'gamma' else None 
    assert col, f"check={check} is NOT supported."

    recs = []
    for i in range(0,5):
        f = 100 if check == 'return' else 1
        row = df[df.week_days==i][col].describe().to_list()
        row = [ e*f for e in row]
        recs += [row]
    weekends_vol = df[(df.week_days==5)|(df.week_days==6)][col].describe().to_list()
    weekends_vol = [ e*100 for e in weekends_vol]
    recs += [ weekends_vol ]
    df = pd.DataFrame.from_records( recs )
    df.columns = ['num','mean','std','min','25%','50%','75%','max']
    df.num /=100

    x = ['' for i in range(0,df.shape[0])]
    x[ last_row.week_days.iloc[0] ] = (last_rtn*100) if check=='return' else last_gamma
    df[f'lastest_{check}'] = f'{x} {last_rtn_rk:.1f}%'
    df['weekday'] = ['mon','tue','wed','thur','fri','weedends']
    df.set_index('weekday',inplace=True,drop=True)

    return df,startts,endts

@click.command()
@click.option('--ric')
@click.option('--rics')
@click.option('--check', default='return', help="return | gamma (derivative of return)")
def main(ric,rics, check):
    if ric:
        df,startts,endts = _main(ric,check)
        print(f'-- {ric} price changes (daily returns%) by *WEEKDAYS*')
        print('-- from', startts, '~', endts)
        print(df)  
    elif rics:
        for ric in rics.split(','):
            df,startts,endts = _main(ric,check)
            print(f'-- {ric} price changes (daily returns%) by *WEEKDAYS*')
            print('-- from', startts, '~', endts)
            print(df) 

if __name__ == '__main__':
    main()