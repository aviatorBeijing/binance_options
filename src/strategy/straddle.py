import os,datetime,json
import pandas as pd 
import click,time
from tabulate import tabulate
import ccxt
import numpy  as np

from butil.butils import DATADIR,get_binance_next_funding_rate,DEBUG
from brisk.bfee import calc_fee

ex = ccxt.binance()

def _find_breakeven(df):
    col = 'net profit @ expiry'
    df['next_neg'] = (df[col]<0).shift(-1) # shift up
    df['prev_neg'] = (df[col]<0).shift(1) # shift down
    df['next_pos'] = (df[col]>0).shift(-1) # shift up
    df['prev_pos'] = (df[col]>0).shift(1) # shift down
    df['is_pos'] = df[col]>0
    df['break_even'] = False; df.loc[ df.prev_pos & df.next_neg, 'break_even'] = True 
    df.loc[df.prev_neg & df.next_pos, 'break_even'] = True 
    df.drop(['next_neg','next_pos','prev_neg','prev_pos','is_pos'], inplace=True, axis=1)
    if not DEBUG:
        df = df[df.break_even]
    return df    

def _v(v): return float(v)
def calc_straddle(  lcontract, rcontract,
                    ldata,rdata, strike_left,strike_right, vol, 
                    taker_order=True, spot_symbol="BTC/USDT",
                    user_premium=0):
    lbid,lask,l_bvol, l_avol = _v(ldata['bid']),_v(ldata['ask']),_v(ldata['bidv']),_v(ldata['askv'])
    rbid,rask,r_bvol, r_avol = _v(rdata['bid']),_v(rdata['ask']),_v(rdata['bidv']),_v(rdata['askv'])
    #assert lask<rask, "Left leg has to be less than right leg (offer price, a.k.a. ask price)"
    print(f'-- order volumes  (P): {vol}-contract, (C): {vol}-contract')
    recs = []
    
    resp = {
            'left': lcontract, 'right': rcontract,
            'is_taker': taker_order, 'paied_premium': user_premium,
            }

    lfee = 0;rfee=0
    if taker_order: # FIXME: Caution, the fee is calculate for varying market prices of options,
                    #        but, if user already holds two positions, fee should be calculated
                    #        against the order prices, instead of market prices. When vol is large,
                    #        the fee may  be significant!
        fee_rate = 5/10000 # Binance: taker 5/10000, maker 2/10000
        premium = (lask + rask)*vol # Assume place instant "taker" orders
        lfee = calc_fee(lask, vol, lcontract, is_taker=True)
        rfee = calc_fee(rask, vol, rcontract, is_taker=True)
        
        print(f'  -- buy Put @ {lask:,.2f} (greeks: {float(ldata["delta"]):.4f}, {float(ldata["gamma"]):.6f},{float(ldata["theta"]):.6f}; iv: {(float(ldata["impvol"])*100):.1f}% )')
        print(f'  -- buy Call @ {rask:,.2f} (greeks: {float(rdata["delta"]):.4f}, {float(rdata["gamma"]):.6f}, {float(rdata["theta"]):.6f}; iv: {(float(rdata["impvol"])*100):.1f}% )')
    else: # maker order (usually hard to fill & sliperage is large.)
        fee_rate = 2/10000
        lfee = calc_fee(lask, vol, lcontract, is_taker=False)
        rfee = calc_fee(rask, vol, rcontract, is_taker=False)
        r = 1 + 5/1000 # A 0.5% higher than current bid price, to enhance chance of getting filled in time.
        premium = (lbid*r + rbid*r)*vol
    
    adhoc = ex.fetch_ticker(spot_symbol)['bid'] # FIXME Binance calc the fee in a DIFFERENT way!
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    
    #fee = vol * adhoc * fee_rate # Binance calc the fee from contract nominal $value.
    #fee *= 2 # put & call
    fee = lfee + rfee
    
    resp['premium'] = premium
    resp['spot'] = adhoc; resp['timestamp'] = ts 
    resp['fee'] = fee 

    liquidation_gain = None # The instant liquidation value of positions
    if user_premium>0: # In case of existing positions, the premium has already been paid.
        # FIXME If specify the user_premium, the trading price of put&call should also be 
        # provided, because they are needed when calc the fee.
        premium = user_premium
        liquidation_value = (lbid + rbid)*vol # instant sell on current  bid price
        liquidation_gain = liquidation_value - user_premium
        liquidation_gain -= fee # FIXME: fee might not be accurate!
        rtn = liquidation_gain/(user_premium+fee)*100
        print(f' '*10,'$'*20, ' Positions ', '$'*20)
        print(' '*15,f'bids (P): ${lbid:.2f}, (C): ${rbid:.2f}; cost: ${(user_premium+fee):.2f}')
        print(' '*15, 'liquidation gain: ', f'${liquidation_gain:.2f}, {rtn:.1f}%')
        print(f' '*10,'$'*53)

    low = adhoc*0.8
    high=adhoc*1.3
    if spot_symbol == 'BTC/USDT':
        low = int(low/1000)*1000
        high = int(high/1000)*1000
        if not DEBUG: step = 100
        else: step = 1000
    elif spot_symbol == 'ETH/USDT':
        low = int(low/100)*100
        high = int(high/100)*100
        if not DEBUG: step = 10
        else: step = 100
    elif spot_symbol == 'BNB/USDT':
        low = int(low/10)*10
        high = int(high/10)*10
        if not DEBUG: step = 1
        else: step = 5
    elif spot_symbol == 'DOGE/USDT':
        step = 0.001
    else:
        raise Exception(f"Unsupported spot symbol: {spot_symbol}.")

    for stock in np.arange(low,high,step): # at expiration
        gains = max(strike_left - stock,0)
        gains += max( stock - strike_right, 0)
        gains *= vol
        profits = gains - premium - fee
        recs += [ ( stock, gains, profits )]
    
    df = pd.DataFrame.from_records( recs, columns=[ f"{spot_symbol} @ expiry",'gain', 'net profit @ expiry'])
    df = _find_breakeven( df )
    cost = premium + fee
    df['stradle_return'] = ( df['net profit @ expiry']) / cost
    df['spot_return'] = (df[f"{spot_symbol} @ expiry"] - adhoc)/adhoc

    #resp['breakeven'] = list(df[f"{spot_symbol} @ expiry"].values)
    resp['break_even'] = list(df['spot_return'].values)

    for col in ['net profit @ expiry']:
        df[col] = df[col].apply(lambda e: f"${e:,.2f}")
    df['stradle_return'] = df['stradle_return'].apply(lambda v: f"{(v*100):.2f}%")
    df['spot_return'] = df['spot_return'].apply(lambda v: f"{(v*100):.1f}%")
    df[f"{spot_symbol} @ expiry"] = df[f"{spot_symbol} @ expiry"].apply(lambda v: f"${v:,.2f}")
    #df.set_index(['spot_return'],inplace=True,drop=True)
    print( tabulate(df, headers="keys"))
    
    print(f'-- spot: ${adhoc:,.2f} @ {ts} (UTC+8)')
    
    print(f'-- order size: {vol} contract  (call&put each)')
    print(f'-- investment  ${premium:,.2f} (premium) + ${fee:,.2f} (fee)')

    return resp 
    

def _main(left,right, vol, is_taker=True, user_premium=0):
    ldata = None;rdata = None
    spot_symbol = left.split('-')[0]+'/USDT'
    annual, funding_rate, ts = get_binance_next_funding_rate( spot_symbol)

    lstrike = left.split('-')[2]
    rstrike = right.split('-')[2]
    strategy = 'Straddle' if lstrike == rstrike else "Strangle"

    print('*'*75)
    print(f'-- funding_rate (perpetual): {(funding_rate*10000):.2f}%%, {(annual*100):.2f}%, {ts}')
    if user_premium>0:
        print(f'-- exiting position premium: ${user_premium}, size: {vol} contract(s)')
    print('*'*75)
    print("-"*10, f' {strategy} Contracts ', '-'*10)
    try:
        with open(f"{DATADIR}/{left.upper()}.json", 'r') as fh:
            ldata = json.loads(fh.read())
            print( '\t',left )#, ldata )
        with open(f"{DATADIR}/{right.upper()}.json", 'r') as fh:
            rdata = json.loads(fh.read())
            print( '\t',right )#, rdata )
    except FileNotFoundError as  e:
        print('*** waiting for data ...')
        time.sleep(5)
        return 
    except json.JSONDecodeError as  je:
        print('*** json data conflict, wait ...')
        time.sleep(5)
        return 
    
    if not ldata:
        raise Exception(f'*** {left.upper()} contract is not found in cached dir: {DATADIR}')
    if not rdata: 
        raise Exception(f'*** {right.upper()} contract is not found in cached dir: {DATADIR}')
    
    strike_left = float(left.split("-")[-2])
    strike_right= float(right.split("-")[-2])
    resp = calc_straddle(  left, right,
                    ldata,rdata, 
                    strike_left,strike_right,
                    vol, 
                    taker_order=is_taker, 
                    spot_symbol = spot_symbol,
                    user_premium=user_premium )
    resp['funding_rate'] = funding_rate
    resp['funding_time'] = ts
    return resp

from multiprocessing import Process
from ws_bcontract import _main as ws_connector

def _multiprocess_main(left,right,vol,user_premium):
    while True:
        try:
            #print('*'*5, "[Taker order]")
            _main(left,right,vol,user_premium=user_premium)
            #print('*'*5, "[Maker order]")
            #_main(left,right,vol, is_taker=False)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--left', help="left leg (OTM put option) contract name")
@click.option('--right', help="right leg (OTM call option)")
@click.option('--size', default=1.0, help="1, 0.1, ... contract size, 1=1BTC contract")
@click.option('--user_premium', default=0., help="a fixed float value, for an existing positions.")
def main(left,right, size,user_premium):

    conn = Process( target=ws_connector, args=(f"{left},{right}", "ticker",) )
    calc = Process( target=_multiprocess_main, args=(left,right,size,user_premium) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()


if __name__ == '__main__':
    main()