import datetime,os,click,time
from multiprocessing import Process 

from ws_bcontract import _main as ws_connector, sync_fetch_ticker

LHISTORY=30 # past 30 seconds
history = [] # bid history
def hadd( new_data:tuple):
    global history
    history += [new_data]
    tnow = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    tnow = int(tnow.timestamp())
    history = filter(lambda e: (tnow-e[0])<LHISTORY, history)

def on_new_market_price( md ):
    ts = md['ts_beijing']
    bid,ask =  md['bid'], md['ask']
    bv,av = md['bidv'], md['askv']

    iv_bid, iv_ask = md['impvol_bid'],md['impvol_ask']
    last = md['last_trade']
    print(  bid,ask,'\t', iv_bid,iv_ask, '\t', bv,av,'\t',  datetime.datetime.fromtimestamp(int(ts))  )

    hadd( (int(ts), float(bid), float(ask), float(bv), float(av), ) )

def _main(contract):
    try:
        while True:
            try:
                sync_fetch_ticker(contract, on_new_market_price )
            except AssertionError as ae:
                print('*** data outdated, wait.', str(ae))
            time.sleep(1)
    except KeyboardInterrupt as ke:
        print('-- shutting down')
        time.sleep(2)

@click.command()
@click.option('--contract')
def main(contract):
    conn = Process( target=ws_connector, args=(f"{contract}", "ticker",) )
    calc = Process( target=_main, args=(contract,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()