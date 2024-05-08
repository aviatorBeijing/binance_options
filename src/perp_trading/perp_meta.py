from sys import is_finalizing
import pandas as pd 
from tabulate import tabulate
import datetime,os
from butil.butils import get_binance_spot

from bbroker.settings import perp_ex

class BinancePerp:
    def __init__(self,ric, ex=None) -> None:
        self.ric = ric 
        self.ex = ex

        # valid price digits
        if ric.startswith('DOGE'): self.ndigits = 5 
        elif ric.startswith('BTC'): self.ndigits = 2
        elif ric.startswith('PENDLE'): self.ndigits = 4
        elif ric.startswith('SOL'): self.ndigits = 2 
        elif ric.startswith('SEI'): self.ndigits = 4
        else:
            raise Exception(f'Unsupported ric: {ric}')
    
    def balance(self)->pd.DataFrame:
        d = self.ex.fetch_balance()
        ttl = d['total'];freed = d['free'];used = d['used']
        _l = lambda d: list(d.items())
        ttl = _l(ttl);freed = _l(freed);used=_l(used)
        ttl = pd.DataFrame.from_records(ttl,columns=['crypto','ttl'],index='crypto')
        freed = pd.DataFrame.from_records(freed,columns=['crypto', 'free'],index='crypto')
        used = pd.DataFrame.from_records(used,columns=['crypto', 'used'],index='crypto')
        df = df[df.ttl>0]
        df = pd.concat([freed,used,ttl],ignore_index=False,axis=1)
        return df

    def check_open_orders(self) -> pd.DataFrame:
        ods = self.ex.fetchOpenOrders(self.ric)
        if not ods:
            print(f'-- no standing orders for ({self.ric})')

        ods = list(map(lambda e: e['info'],ods))
        df = pd.DataFrame.from_records(ods)
        if df.empty: 
            print('*** No outstanding orders.')
            return pd.DataFrame()
        #df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df['symbol,type,side,status,orderId,price,origQty,executedQty,updateTime'.split(',')] #cummulativeQuoteQty
        df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
        df = df.sort_values('updateTime', ascending=False)
        bid,ask = adhoc_ticker(self.ric.upper())
        df = BinancePerp.est_pnl_on_open_orders(df, (bid+ask)*.5)
        
        print('--[ orders ]\n',tabulate(df,headers="keys"))
        fn = os.getenv("USER_HOME","/Users/junma")
        fn += f'/tmp/binance_perp_open_orders_{self.ric.lower().replace("/","-")}.csv'
        df.to_csv( fn, index=0)
        print('-- saved:', fn)
        return df  
    
    @staticmethod
    def est_pnl_on_open_orders(df,p0)->pd.DataFrame:
        """
        @param p0: reference current spot price
        """
        df = df.copy()
        df['sign'] = df.side.apply(lambda s: -1 if s=='SELL' else 1  if s=='BUY' else 0)
        df['$loss'] = (df.price.astype(float) - p0) * df.sign * df.origQty.astype(float)
        df['drift_bps'] = ((df.price.astype(float)-p0)/p0*10_000).apply(int)
        df.drop(['sign'],axis=1,inplace=True)
        return df 

    @staticmethod
    def analyze_open_orders_cached(p0,ric) ->pd.DateOffset:
        fn = os.getenv("USER_HOME","/Users/junma")
        fn += f'/tmp/binance_open_orders_{ric.lower().replace("/","-")}.csv'
        df = pd.read_csv( fn )
        df['datetime'] = df['datetime'].apply(pd.Timestamp)

        df = BinancePerp.est_pnl_on_open_orders(df,p0)

        print(f'-- [open orders from cached file: {fn}]')
        print( tabulate(df, headers='keys') )
        return df

    def check_trades_today(self)  -> pd.DataFrame:
        return self.check_trades(hours=24)
    
    def check_trades(self, hours=24):
        dt = datetime.datetime.utcnow().timestamp() - hours*3600
        dt = int(dt)*1000
        tds = self.ex.fetchMyTrades(self.ric, since=dt, limit=None, params={})
        tds = list(map(lambda e: e['info'],tds))
        df = pd.DataFrame.from_records(tds)
        print('$'*50,'\n', df)
        if df.empty: 
            print('*** No outstanding orders.')
            return pd.DataFrame()
        #df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df['id,symbol,qty,price,commission,commissionAsset,maker,buyer,time'.split(',')]
        df['side'] = df.isBuyer.apply(lambda v: 'SELL' if not v else "BUY")
        df = df['id,symbol,side,qty,price,commission,commissionAsset,maker,time'.split(',')]
        df['datetime'] = df.time.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
        df = df.sort_values('time', ascending=True)
        #print('--[ trades in 24 hours ]\n',tabulate(df,headers="keys"))
        return df  

    def buy(self,price,qty,ask):
        """
        @param ask: must pass in the current ask price on order book

        {'info': {'symbol': 'DOGEUSDT', 
                'orderId': '4888578705', 
                'orderListId': '-1', 
                'clientOrderId': 'x-R4BD3S8268d769b85f2481f7d68115', 
                'transactTime': '1713079091348', 
                'price': '0.14000000', 
                'origQty': '50.00000000', 
                'executedQty': 
                '0.00000000', 
                'cummulativeQuoteQty': '0.00000000', '
                status': 'NEW', 'timeInForce': 'GTC', 
                'type': 'LIMIT', 'side': 'BUY', 'workingTime': '1713079091348', 'fills': [], 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '4888578705', 'clientOrderId': 'x-R4BD3S8268d769b85f2481f7d68115', 'timestamp': 1713079091348, 'datetime': '2024-04-14T07:18:11.348Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1713079091348, 'symbol': 'DOGE/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.14, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'open', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None}
        """
        assert price<ask, f"buying price must be less than ask price. price={price}, ask={ask}"
        sym = self.ric.replace('-','/').upper()
        res = self.ex.createLimitBuyOrder(sym,qty,price,params={})
        print(res['info']['orderId'])
    
    def sell(self,price,qty,bid):
        """
        @param bid: must pass in the current bid price on order book
        """
        assert price>bid, f"selling price must be greater than ask price. price={price}, bid={bid}"
        sym = self.ric.replace('-','/').upper()
        res = self.ex.createLimitSellOrder(sym,qty,price,params={})
        print(res['info']['orderId'])

    def cancel_order(self, oid:str): # cancel single order
        """
        {'info': {'symbol': 'DOGEUSDT', 
            'origClientOrderId': 'x-R4BD3S8268d769b85f2481f7d68115', 
            'orderId': '4888578705', 'orderListId': '-1', 
            'clientOrderId': '8zE1oB4p40vUVoSsayL3w9', 
            'transactTime': '1713079742955', 'price': '0.14000000', 'origQty': '50.00000000', 
            'executedQty': '0.00000000', 
            'cummulativeQuoteQty': '0.00000000', 
            'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '4888578705', 'clientOrderId': '8zE1oB4p40vUVoSsayL3w9', 'timestamp': 1713079742955, 'datetime': '2024-04-14T07:29:02.955Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1713079742955, 'symbol': 'DOGE/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.14, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None}

        """
        if oid:
            res = self.ex.cancelOrder( oid, symbol=self.ric.replace('-','/') )
            assert res['info']['status'] == 'CANCELED', f'oid={oid} failed to cancel. resp: \n{res}'
            rqty = float( res['info']['executedQty'])
            if rqty>0:
                print(f'  -- oid={oid} canceled ok')
                print(f'  -- oid={oid} filled before cancelation')

def main_(ex, cbuy,csell,price,qty,sellbest,buybest):
    bid,ask = adhoc_ticker(ex.ric)
    print(f'-- current bid={bid}, ask={ask}; requesting: price={price}, qty={qty}, {"sell" if csell else "buy" if cbuy else "unknown"}')
    if cbuy:
        ex.buy(price,qty,ask)
    elif csell:
        ex.sell(price,qty,bid)
    elif buybest:
        pce = bid * (1-1/10_000.)
        #pce = round(pce, ex.ndigits)
        ex.buy(pce,qty,ask)
    elif sellbest:
        pce = ask * (1+1/10_000.)
        #pce = round(pce, ex.ndigits)
        ex.sell(pce,qty,bid)
    else:
        print('*** nothing to do.')


def adhoc_ticker(symbol='BTC/USDT')->tuple:
    symbol = symbol.replace('-','/')
    #qts = perp_ex.fetch_ticker(symbol)
    qts = perp_ex.public_get_ticker_bookticker({'symbol': symbol.replace('/','').replace('-','').upper()})
    bid,ask = qts['bidPrice'],qts['askPrice'] # bidQty,askQty
    return float(bid),float(ask)

import click
@click.command()
@click.option('--ric')
@click.option('--check',is_flag=True, default=False)
@click.option('--cbuy', is_flag=True,  default=False)
@click.option('--csell', is_flag=True,  default=False)
@click.option('--cancel', default='', help='comma-separated order ids to be canceled')
@click.option('--price',default=0.)
@click.option('--qty',default=0.)
@click.option('--sellbest', is_flag=True, default=False,help='judge from ask price, automatic create an order close to ask price')
@click.option('--buybest',  is_flag=True, default=False,help='judge from bid price, automatic create an order close to ask price')
@click.option('--centered_pair', is_flag=True, default=False, help='generate a pair of orders set apart by 100bsp around the bid/ask')
@click.option('--centered_pair_dist', default=50., help='generate a pair of orders set apart by # (bps) around the bid/ask')
@click.option('--buyup', default=0., help='use the best price to buy the quantity, simultaneously sell same qty at 50bps up')
@click.option('--selldown', default=0., help='use the best price to sell the quantity, simultaneously buy same qty at 50bps down')
def main(ric,check,cbuy,csell,cancel,price,qty,sellbest,buybest,centered_pair,centered_pair_dist,buyup,selldown):
    assert 'USDT' in ric, r'Unsuported: {ric}'
    assert '-' in ric or '/' in ric, r'Unsupported: {ric}, use "-" or "/" in ric name'
    ex = BinancePerp(ric.replace('-','/'), ex=perp_ex)

    if check:
        ex.check_open_orders() 
        ex.check_trades_today()
    elif cancel:
        for oid in cancel.split(','):
            ex.cancel_order( oid )
    elif centered_pair:
        assert qty>0, 'Must provide a qty>0'
        bid,ask = adhoc_ticker(ric)
        spread = (ask-bid)/(ask+bid)*2
        assert spread< 5./10_000, f'spread is too wide: {spread} (bid:{bid},ask:{ask})'
        pce = (bid+ask)*.5
        assert centered_pair_dist > 20, f"{centered_pair_dist} is too low, suggest to > 20 or 50"
        e = float(centered_pair_dist)/10_000.
        buy_pce = pce*(1-e)
        sell_pce= pce*(1+e)
        print('-- price diff:', sell_pce-buy_pce)
        ex.buy( buy_pce, qty, ask ) # Buy relativly lower
        ex.sell(sell_pce, qty, bid)
    elif sellbest or buybest:
        assert qty>0, f"qty is required"
        main_(ex,False,False,0.,qty,sellbest,buybest)
    elif buyup>0:
        bid,ask = adhoc_ticker(ric);spread = (ask-bid)/(ask+bid)*2
        assert spread< 5./10_000, f'spread is too wide: {spread} (bid:{bid},ask:{ask})'
        ex.buy(bid,buyup,ask)
        ex.sell(ask*(1.+50./10_000),buyup,bid)
    elif selldown>0:
        bid,ask = adhoc_ticker(ric);spread = (ask-bid)/(ask+bid)*2
        assert spread< 5./10_000, f'spread is too wide: {spread} (bid:{bid},ask:{ask})'
        ex.sell(ask,selldown,bid)
        ex.buy(bid*(1.-50./10_000),selldown,ask)
    else:
        price = float(price)
        qty = float(qty)
        main_(ex,cbuy,csell,price,qty,sellbest,buybest)

if __name__ == '__main__':
    main()
