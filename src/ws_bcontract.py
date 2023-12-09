import datetime,click
import websocket
import _thread
import time
import rel,json
import pandas as pd

dedups = {}

def on_message(ws, message):
    msg = json.loads( message )
    fds = ['s','c', 'mp', 'bo','ao','bq','aq', 'b','a','d','g','t','vo','V','A'] #for 'ticker'
    df = pd.DataFrame.from_records([ msg ] )
    #print( df[ fds ] )

    df['spread'] = df['ao'].astype(float)-df['bo'].astype(float)
    rows = df[['s','c', 'bo','ao', 'spread']].to_records(index=False)
    for row in rows:
        row = list(row)
        sym = row[0]; is_updating = False
        val = ','.join(row[1:len(row)-1] )
        if sym not in dedups:
            dedups[ sym ] = val
            is_updating = True
        else:
            if dedups[sym] != val:
                dedups[sym] = val
                is_updating = True 
        if is_updating:
            print( sym, 'trade|bid|ask|spread', row[1:] )

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
