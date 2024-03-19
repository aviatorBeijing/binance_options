import datetime,os,click,time
from multiprocessing import Process 

from ws_bcontract import _main as ws_connector, sync_fetch_ticker

def on_new_market_price( md ):
    bid,ask =  md['bid'], md['ask']
    bv,av = md['bidv'], md['askv']
    last = md['last_trade']
    ts = md['ts_beijing'];ts=datetime.datetime.fromtimestamp(int(ts))
    print(  bid,ask,bv,av,last,ts  )

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