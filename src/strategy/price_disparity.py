import os,datetime,click,time,json,ccxt
import pandas as pd 
from multiprocessing import Process
import numpy as np
import ccxt
ex = ccxt.binance()

from ws_bcontract import _main as ws_connector, _maturity
from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from strategy.delta_gamma import callprice,putprice

def get_maturity(contract):
    fds = contract.split('-')
    ts = datetime.datetime.strptime('20'+fds[1], '%Y%m%d') + datetime.timedelta(hours=8) # Settle at 08:00 (UTC) of the date.
    tnow = datetime.datetime.utcnow()
    dt = (ts-tnow).total_seconds()/3600./24
    return dt

def extract_specs( contract):
    sym,expiry,strike,typ = contract.split('-')
    maturity = get_maturity( contract )
    strike = float(strike)
    ctype = 'call' if typ=='C' else 'put'
    spot = f'{sym.upper()}/USDT'
    return spot, maturity, strike, ctype
    
def check_disparity(contract,market_df):
    underlying,T,K,ctype = extract_specs( contract)
    spot_price = ex.fetch_ticker(underlying)['bid']
    
    print( market_df )
    market_quote_bid = market_df.iloc[0].bid
    market_quote_ask = market_df.iloc[0].ask
    market_impvol = market_df.iloc[0].impvol
    market_impvol_bid = market_df.iloc[0].impvol_bid
    market_impvol_ask = market_df.iloc[0].impvol_ask

    def _diff(a,b):
        x = (a-b)/b*100
        return f"{x:.1f}%"
    recs = []
    sigmas = np.arange(10/100, 150/100, 1/100)
    interests = np.arange(-5/100, 5/100, 1/100)
    for sigma in sigmas:
        for r in interests: # risk-free rate
            if ctype == 'call':
                option_price = callprice(spot_price, K, T/365, sigma, r )
            elif ctype == 'put':
                option_price =  putprice(spot_price, K, T/365, sigma, r )
            recs += [ (contract, r, sigma, option_price, 
                            market_quote_bid-option_price, market_quote_ask-option_price,
                            _diff(market_impvol,sigma), 
                            _diff(market_impvol_bid,sigma), 
                            _diff(market_impvol_ask,sigma) ) ]
    df = pd.DataFrame.from_records(recs, columns=[
        'contract', 'rf', 'bsm_vol', 'bsm_fair', 'bid-bsm','ask-bsm','impvol','impvol_bid', 'impvol_ask'
    ])
    df = df[ (abs(df['bid-bsm'])<2) | (abs(df['ask-bsm'])<2)]
    df.bsm_fair = df.bsm_fair.apply(lambda v: (int(v*10)/10))
    df.sort_values(['bid-bsm','ask-bsm'], inplace=True, ascending=False)
    print( df )

def _main( contracts ):
    try:
        recs = []
        for contract in contracts:
            fn = f"{DATADIR}/{contract.upper()}.json"
            print('-- reading:', fn)
            with open(fn, 'r') as fh:
                contract_data = json.loads(fh.read())
                contract_data['c'] = contract
            df = pd.DataFrame.from_records( [ contract_data ] )

            df.reset_index(inplace=True,drop=True)
            for col in df.columns[ :df.shape[1]-1]:
                df[col] = df[col].apply(float)
            check_disparity(contract, df)

    except FileNotFoundError as  e:
        print('*** waiting for data ...')
        time.sleep(5)
        return 
    except json.JSONDecodeError as  je:
        print('*** json data conflict, wait ...')
        time.sleep(5)
        return

def _mp_main(contracts):
    while True:
        try:
            _main(contracts)
            print('\n\n')
            time.sleep(10)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contracts')
def main(contracts):
    conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    calc = Process( target=_mp_main, args=(contracts.split(","), ) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()
