import datetime,click
import websocket
import _thread
import time,os
import rel,json
import pandas as pd

DATADIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/options'
if not os.path.exists( DATADIR):
    os.makedirs( DATADIR )
print('-- data dir:', DATADIR)

def _maturity( symbol ):
    ds = symbol.split('-')[1]
    t = datetime.datetime.strptime( '20'+ds, '%Y%m%d')
    tnow = datetime.datetime.utcnow()
    dt = t - tnow
    return f"{dt.days}d {dt.seconds//3600}h {(dt.seconds//60)%60}m"

dedups = {}

class MaxVolatility:
    def __init__(self, symbol, vol):
        self.symbol = symbol
        self.vol = vol

    def __str__(self):
        return f"-- {self.symbol}, vol={self.vol}"
    def __expr__(self):
        return self.__str__()
max_volatility = None

def on_message(ws, message):
    global max_volatility
    msg = json.loads( message )
    fds = ['s','c', 'mp', 'bo','ao','bq','aq', 'b','a','d','g','t','vo','V','A'] #for 'ticker'
    df = pd.DataFrame.from_records([ msg ] )
    #print( df[ fds ] )

    avg = ( df['ao'].astype(float)+df['bo'].astype(float) )/2
    df['spread'] = df['ao'].astype(float)-df['bo'].astype(float)
    
    df['spd%'] = df['spread']/avg
    df['spd%'] = df['spd%'].apply(lambda v: f"{(v*100):.1f}%")
    
    vo = float(df.vo.values[0] )
    if max_volatility:
        old_v = max_volatility.vol
        if old_v < vo:
            max_volatility.symbol = df['s'].values[0]
            max_volatility.vol = vo
            print( max_volatility)
    else:
        max_volatility = MaxVolatility(df['s'].values[0], vo)

    rows = df[['s','c', 'bo','ao', 'spread','spd%']].to_records(index=False)
    for row in rows:
        row = list(row)
        sym = row[0]; is_updating = False
        val = ','.join(row[1:len(row)-2] )
        if sym not in dedups:
            dedups[ sym ] = val
            is_updating = True
        else:
            if dedups[sym] != val:
                dedups[sym] = val
                is_updating = True 
        if is_updating:
            m = _maturity( sym )
            #BTC-240927-60000-C ['5645', '4005', '6415', 2410.0]
            print( sym, m, 'trade|bid|ask|spread|spd%', row[1:] )
            with open(f"{DATADIR}/{sym}.json", 'w') as fh:
                data = {
                        "last_trade": df.iloc[0].c,
                        "bid": df.iloc[0].bo,
                        "ask": df.iloc[0].ao,
                        "bidv": df.iloc[0].bq,
                        "askv": df.iloc[0].aq}
                json.dump(data, fh)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

endpoint = 'wss://nbstream.binance.com/eoptions/ws/{symbol}@{channel}' #trade|ticker

@click.command()
@click.option('--rics')
@click.option('--channel', default="trade")
def main(rics, channel):
    print('-- channel:', channel)
    rics = rics.split(',')
    uris = list(map(lambda ric: endpoint.format( symbol=ric, channel=channel), rics ) )
    websocket.enableTrace(False) #True)
    for uri in uris:
        ws = websocket.WebSocketApp(uri, #wss://api.gemini.com/v1/marketdata/BTCUSD",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

        ws.run_forever(dispatcher=rel, reconnect=5)  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()

if __name__ == '__main__':
    main()
