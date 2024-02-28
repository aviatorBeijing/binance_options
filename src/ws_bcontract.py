import datetime,click
import websocket
import time,os
import rel,json
import pandas as pd

from butils import DATADIR,DEBUG

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
    #print( len(msg), msg)
    df = pd.DataFrame.from_records([ msg ] )
    #print( df[ fds ] )

    avg = ( df['ao'].astype(float)+df['bo'].astype(float) )/2
    df['spread'] = df['ao'].astype(float)-df['bo'].astype(float)
    
    df['delta'] = df['d']
    df['gamma'] = df['g']
    df['theta'] = df['t']
    df['vega']  = df['v']
    df['impvol'] = df['vo']

    df['spd%'] = df['spread']/avg
    df['spd%'] = df['spd%'].apply(lambda v: f"{(v*100):.1f}%")
    
    vo = float(df.vo.values[0] )
    if max_volatility:
        old_v = max_volatility.vol
        if old_v < vo:
            max_volatility.symbol = df['s'].values[0]
            max_volatility.vol = vo
            if DEBUG:
                print( '-- new max vol:', max_volatility)
    else:
        max_volatility = MaxVolatility(df['s'].values[0], vo)
    
    rows = df[['s','c', 'bo','ao', 'spread','spd%', 'delta','gamma','theta','vega','impvol']].to_records(index=False)
    for row in rows:
        row = list(row)
        sym = row[0]; is_updating = False
        val = ','.join(row[1:4] )
        if sym not in dedups:
            dedups[ sym ] = val
            is_updating = True
        else:
            if dedups[sym] != val:
                dedups[sym] = val
                is_updating = True 
        if is_updating:
            m = _maturity( sym )
            if DEBUG:
                print( sym, m, 'trade|bid|ask|spread|spd%', row[1:] )
            with open(f"{DATADIR}/{sym}.json", 'w') as fh:
                data = {
                        "last_trade": df.iloc[0].c,
                        "bid": df.iloc[0].bo,
                        "ask": df.iloc[0].ao,
                        "bidv": df.iloc[0].bq,
                        "askv": df.iloc[0].aq,
                        "delta": df.iloc[0].delta,
                        "gamma": df.iloc[0].gamma,
                        "theta": df.iloc[0].theta,
                        "vega": df.iloc[0].vega,
                        "impvol": df.iloc[0].impvol
                        }
                json.dump(data, fh)
                

def on_error(ws, error):
    print('error:', error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

endpoint = 'wss://nbstream.binance.com/eoptions/ws/{symbol}@{channel}' #trade|ticker

def _main(rics:str, channel):
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

@click.command()
@click.option('--rics')
@click.option('--channel', default="trade")
def main(rics, channel):
    print('-- channel:', channel)
    _main(rics, channel)

if __name__ == '__main__':
    main()
