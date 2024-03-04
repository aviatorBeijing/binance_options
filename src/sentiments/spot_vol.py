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

def _main(ric):
    if not os.path.exists('btc.csv'):
        endts = (datetime.datetime.utcnow()+datetime.timedelta(hours=8)).timestamp()
        startts = endts - 5*365*24*3600
        resp = ydata( ric, startts,endts)
        timestamps = resp['chart']['result'][0]['timestamp']
        data = resp['chart']['result'][0]['indicators']['quote'][0]
        
        df = pd.DataFrame.from_dict({"date": timestamps, 
                                    "open": data['open'],
                                    "high": data['high'],
                                    "low": data['low'],
                                    "close": data['close'] })
        df.to_csv("btc.csv", index=False)
    else:
        df = pd.read_csv('btc.csv')
        startts = df.date.iloc[0]
        endts = df.date.iloc[-1]
    
    df.date = df.date.apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
    #df.set_index('date',drop=True,inplace=True)
    df['week_days']= df.date.apply(lambda e: e.dayofweek)
    
    df.dropna(inplace=True)
    df['rtn'] = df.close.pct_change()
    df['gamma'] = df.rtn.pct_change()

    recs = []
    for i in range(0,5):
        row = df[df.week_days==i].rtn.describe().to_list()
        row = [ e*100 for e in row]
        recs += [row]
    weekends_vol = df[(df.week_days==5)|(df.week_days==6)].rtn.describe().to_list()
    weekends_vol = [ e*100 for e in weekends_vol]
    recs += [ weekends_vol ]
    df = pd.DataFrame.from_records( recs )
    df.columns = ['num','mean','std','min','25%','50%','75%','max']
    df.num /=100

    df['weekday'] = ['mon','tue','wed','thur','fri','weedends']
    df.set_index('weekday',inplace=True,drop=True)

    print('-- BTC/USD price changes (daily returns%) by *WEEKDAYS*')
    print('-- from', datetime.datetime.fromtimestamp(startts), '~', datetime.datetime.fromtimestamp(endts))
    print(df)    

@click.command()
@click.option('--ric')
def main(ric):
    _main(ric)
if __name__ == '__main__':
    main()