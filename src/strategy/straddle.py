import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import ccxt
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG,
                get_binance_next_funding_rate,
                get_maturity )
from brisk.bfee import calc_fee
from strategy.price_disparity import _main as check_bsm_disparity
from ws_bcontract import _main as ws_connector
from strategy.delta_gamma import callprice,putprice

ex = ccxt.binance()

from strategy.delta_gamma import fair_call_vol,fair_put_vol

def _find_breakeven(df, adhoc, epsilon=5, user_premium=0.):
    col = 'net profit @ expiry'
    df['next_neg'] = (df[col]<0).shift(-1) # shift up
    df['prev_neg'] = (df[col]<0).shift(1) # shift down
    df['next_pos'] = (df[col]>0).shift(-1) # shift up
    df['prev_pos'] = (df[col]>0).shift(1) # shift down
    df['is_pos'] = df[col]>0
    df['break_even'] = False; df.loc[ df.prev_pos & df.next_neg, 'break_even'] = True 
    df.loc[df.prev_neg & df.next_pos, 'break_even'] = True 
    df.drop(['next_neg','next_pos','prev_neg','prev_pos','is_pos'], inplace=True, axis=1)

    # find current spot position
    df['sd'] = df.iloc[:,0] - adhoc 
    df['spot_vicinity'] = df.sd.apply(abs) < epsilon
    df.drop(['sd'], axis=1, inplace=True)

    if user_premium >0:
        df['call_break_even'] = df['bsm_call'] - user_premium
        #df['call_break_even'] = df['call_break_even'].apply(abs)
        df['call_break_even'] =  df['call_break_even'] *  df['call_break_even'].shift(1)
        df['call_break_even'] = df['call_break_even']<0

        df['put_break_even'] = df['bsm_put'] - user_premium
        df['put_break_even'] =  df['put_break_even'] *  df['put_break_even'].shift(1)
        df['put_break_even'] = df['put_break_even']<0
    
    if not DEBUG:
        if user_premium >0:
            df = df[ (df.break_even) | (df.spot_vicinity) | (df.call_break_even) | (df.put_break_even) ]
        else:
            df = df[ (df.break_even) | (df.spot_vicinity) ]
    return df    

def _v(v): return float(v)
def calc_straddle(  lcontract, rcontract,
                    ldata,rdata, strike_left,strike_right, vol, 
                    taker_order=True, spot_symbol="BTC/USDT",
                    user_premium=0):
    lbid,lask,l_bvol, l_avol = _v(ldata['bid']),_v(ldata['ask']),_v(ldata['bidv']),_v(ldata['askv'])
    rbid,rask,r_bvol, r_avol = _v(rdata['bid']),_v(rdata['ask']),_v(rdata['bidv']),_v(rdata['askv'])
    if lask ==0 or rask == 0:
        return {}

    #assert lask<rask, "Left leg has to be less than right leg (offer price, a.k.a. ask price)"
    print(f'-- order volumes  (P): {vol}-contract, (C): {vol}-contract')
    recs = []
    
    adhoc = S = ex.fetch_ticker(spot_symbol)['bid'] # FIXME Binance calc the fee in a DIFFERENT way!
    
    resp = {
            'left': lcontract, 'right': rcontract,
            'is_taker': taker_order, 'paid_premium': user_premium,
            }

    # Maturity and time-values left
    lmaturity = Tl = get_maturity( lcontract )
    rmaturity = Tr = get_maturity( rcontract )
    timeValueL = -float(ldata["theta"]) * Tl 
    timeValueR = -float(rdata["theta"]) * Tr
    timeValueLPct = timeValueL/lask * 100
    timeValueRPct = timeValueR/rask * 100 

    resp['time_values'] = {}
    resp['time_values'][lcontract] = timeValueL
    resp['time_values'][rcontract] = timeValueR

    # vols
    K = float(lcontract.split('-')[2])
    rf = 0.
    if lcontract.endswith('-C'):
        fairvol_bid = fair_call_vol(float(ldata["bid"]),S,K,Tl/365, r=rf)
        fairvol_ask = fair_call_vol(float(ldata["ask"]),S,K,Tl/365, r=rf)
    elif lcontract.endswith('-P'):
        fairvol_bid = fair_put_vol(float(ldata["bid"]),S,K,Tl/365, r=rf)
        fairvol_ask = fair_put_vol(float(ldata["ask"]),S,K,Tl/365, r=rf)
    print(f'  -- bsm vols of {lcontract} (bid,ask): {(fairvol_bid):.1f}% ({(float(ldata["impvol_bid"])*100):.1f}%), {(fairvol_ask):.1f}% ({(float(ldata["impvol_ask"])*100):.1f}%)')

    K = float(rcontract.split('-')[2])
    if rcontract.endswith('-C'):
        fairvol_bid = fair_call_vol(float(rdata["bid"]),S,K,Tr/365, r=rf)
        fairvol_ask = fair_call_vol(float(rdata["ask"]),S,K,Tr/365, r=rf)
    elif rcontract.endswith('-P'):
        fairvol_bid = fair_put_vol(float(rdata["bid"]),S,K,Tr/365, r=rf)
        fairvol_ask = fair_put_vol(float(rdata["ask"]),S,K,Tr/365, r=rf)
    print(f'  -- bsm vols of {rcontract} (bid,ask): {(fairvol_bid):.1f}% ({(float(rdata["impvol_bid"])*100):.1f}%), {(fairvol_ask):.1f}% ({(float(rdata["impvol_ask"])*100):.1f}%)')


    lfee = 0;rfee=0
    if taker_order: # FIXME: Caution, the fee is calculate for varying market prices of options,
                    #        but, if user already holds two positions, fee should be calculated
                    #        against the order prices, instead of market prices. When vol is large,
                    #        the fee may  be significant!
        fee_rate = 5/10000 # Binance: taker 5/10000, maker 2/10000
        premium = (lask + rask)*vol # Assume place instant "taker" orders
        lfee = calc_fee(lask, vol, lcontract, is_taker=True)
        rfee = calc_fee(rask, vol, rcontract, is_taker=True)
        
        print(f'  -- buy {lcontract} @ {lask:,.2f} (greeks: {float(ldata["delta"]):.4f}, {float(ldata["gamma"]):.6f},{float(ldata["theta"]):.6f}, {(float(ldata["impvol"])*100):.1f}% ); time-value left: ${timeValueL:.2f}, {timeValueLPct:.1f}%')
        print(f'  -- buy {rcontract} @ {rask:,.2f} (greeks: {float(rdata["delta"]):.4f}, {float(rdata["gamma"]):.6f}, {float(rdata["theta"]):.6f}, {(float(rdata["impvol"])*100):.1f}% ); time-value left: ${timeValueR:.2f}, {timeValueRPct:.1f}%')
    else: # maker order (usually hard to fill & sliperage is large.)
        fee_rate = 2/10000
        lfee = calc_fee(lask, vol, lcontract, is_taker=False)
        rfee = calc_fee(rask, vol, rcontract, is_taker=False)
        r = 1 + 5/1000 # A 0.5% higher than current bid price, to enhance chance of getting filled in time.
        premium = (lbid*r + rbid*r)*vol
    
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    
    #fee = vol * adhoc * fee_rate # Binance calc the fee from contract nominal $value.
    #fee *= 2 # put & call
    fee = lfee + rfee
    
    resp['premium'] = premium
    resp['fee'] = fee 
    resp['spot'] = adhoc#; resp['timestamp'] = ts 

    liquidation_gain = None # The instant liquidation value of positions
    if user_premium>0: # In case of existing positions, the premium has already been paid.
        # FIXME If specify the user_premium, the trading price of put&call should also be 
        # provided, because they are needed when calc the fee.
        premium = user_premium
        liquidation_value = (lbid + rbid)*vol # instant sell on current  bid price
        liquidation_gain = liquidation_value - user_premium
        liquidation_gain -= fee # FIXME: fee might not be accurate!
        rtn = liquidation_gain/(user_premium+fee)*100
        ldelta = _v(ldata['delta']);rdelta = _v(rdata['delta'])
        print(f' '*10,'$'*20, ' Positions ', '$'*20)
        print(' '*15,f'spot (now): ${adhoc}' )
        print(' '*15,f'position delta: {(ldelta+rdelta):.3f}')
        print(' '*15,f'         theta: {_v(ldata["theta"])}, {_v(rdata["theta"])}')
        print(' '*15,f'bids (P): ${lbid:.2f}, (C): ${rbid:.2f}; cost: ${(user_premium+fee):.2f}')
        print(' '*15, 'liquidation gain: ', f'${liquidation_gain:.2f}, {rtn:.1f}%')
        print(f' '*10,'$'*53)

    low = adhoc*0.7
    high=adhoc*1.5
    epsilon=5
    if not DEBUG:
        low = adhoc*0.85
        high = adhoc*1.15
    face = 1
    if spot_symbol == 'BTC/USDT':
        low = int(low/1000)*1000
        high = int(high/1000)*1000
        if not DEBUG: step = 5
        else: step = 1000
    elif spot_symbol == 'ETH/USDT':
        low = int(low/100)*100
        high = int(high/100)*100
        if not DEBUG: step = 1
        else: step = 100
    elif spot_symbol == 'BNB/USDT':
        low = int(low/10)*10
        high = int(high/10)*10
        if not DEBUG: step = 1
        else: step = 5
    elif spot_symbol == 'DOGE/USDT':
        low = adhoc*0.75
        high = adhoc*1.25
        step = 0.001
        face = 1000
        epsilon=0.001
    else:
        raise Exception(f"Unsupported spot symbol: {spot_symbol}.")

    for stock in np.arange(low,high,step): # at expiration
        gains = max(strike_left - stock,0)
        gains += max( stock - strike_right, 0)
        gains *= vol*face
        profits = gains - premium - fee
        recs += [ ( stock, gains, profits )]
    
    sym = f"{spot_symbol} @ expiry"
    df = pd.DataFrame.from_records( recs, columns=[ sym,'gain', 'net profit @ expiry'])

    # BSM prices on changing Spot
    lvol = float(ldata["impvol"])
    rvol = float(rdata["impvol"])
    df['bsm_call'] =  df[sym].apply( lambda S: callprice(S,strike_right,rmaturity/365, rvol, 0) )
    df['bsm_put'] =  df[sym].apply( lambda S: putprice(S,strike_left,lmaturity/365, lvol, 0) )
    
    df = _find_breakeven( df, adhoc, epsilon, user_premium/vol )

    cost = premium + fee
    df['stradle_return'] = ( df['net profit @ expiry']) / cost
    df['spot_return'] = (df[f"{spot_symbol} @ expiry"] - adhoc)/adhoc

    #resp['breakeven'] = list(df[f"{spot_symbol} @ expiry"].values)
    resp['be_returns'] = list( map(lambda v: int(v*1e4)/1e4, df['spot_return'].values) )
    resp['be_prices'] = list(df[f"{spot_symbol} @ expiry"].values)
    resp['straddle_returns'] = list( map(lambda v: int(v*1e4)/1e4, df['stradle_return'].values) )

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

    return resp, df
    

def _main(left,right, vol, is_taker=True, user_premium=0,check_parity=False):
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
    
    """try:
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
    """
    ldata = fetch_bidask(left.upper())
    rdata = fetch_bidask(right.upper())
    
    if not ldata:
        raise Exception(f'*** {left.upper()} contract is not found in cached dir: {DATADIR}')
    if not rdata: 
        raise Exception(f'*** {right.upper()} contract is not found in cached dir: {DATADIR}')
    
    if check_parity:
        check_bsm_disparity( [left,right] )

    strike_left = float(left.split("-")[-2])
    strike_right= float(right.split("-")[-2])
    resp, df = calc_straddle(  left, right,
                    ldata,rdata, 
                    strike_left,strike_right,
                    vol, 
                    taker_order=is_taker, 
                    spot_symbol = spot_symbol,
                    user_premium=user_premium )
    if resp: #non-empty
        resp['funding_rate'] = funding_rate
        resp['funding_time'] = ts
    else: pass # keep it empty
    return resp

def _multiprocess_main(left,right,vol,user_premium,check_parity):
    print('-- waiting data...')
    time.sleep(2)
    while True:
        try:
            #print('*'*5, "[Taker order]")
            _main(left,right,vol,user_premium=user_premium,check_parity=check_parity)
            #print('*'*5, "[Maker order]")
            #_main(left,right,vol, is_taker=False)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--left', help="left leg (OTM put option) contract name")
@click.option('--right', help="right leg (OTM call option)")
@click.option('--contracts', help="a pair of put/call contracts, comma-separated")
@click.option('--size', default=1.0, help="1, 0.1, ... contract size, 1=1BTC contract")
@click.option('--user_premium', default=0., help="a fixed float value, for an existing positions.")
@click.option('--check_parity', is_flag=True, default=False)
def main(left,right,contracts,size,user_premium, check_parity):
    
    if contracts:
        print('-- ignoring "--left" and "--right", using "--contracts". ')
        left = list(filter(lambda s: s.endswith('P'), contracts.split(',')))[0]
        right = list(filter(lambda s: s.endswith('C'), contracts.split(',')))[0]
    
    conn = Process( target=ws_connector, args=(f"{left},{right}", "ticker",) )
    calc = Process( target=_multiprocess_main, args=(left,right,size,user_premium,check_parity) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()
    #_multiprocess_main(left,right,size,user_premium,check_parity)



if __name__ == '__main__':
    main()
