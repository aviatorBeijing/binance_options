from sys import is_finalizing
import pandas as pd 
from tabulate import tabulate
import datetime,os
class BianceSpot:
    def __init__(self,ric, spot_ex=None) -> None:
        self.ric = ric 
        self.ex = spot_ex

        # valid price digits
        if ric.startswith('DOGE'): self.ndigits = 5 
        elif ric.startswith('BTC'): self.ndigits = 2
        else:
            raise Exception(f'Unsupported ric: {ric}')

    def check_open_orders(self) -> pd.DataFrame:
        ods = self.ex.fetchOpenOrders(self.ric)
        ods = list(map(lambda e: e['info'],ods))
        df = pd.DataFrame.from_records(ods)
        if df.empty: 
            print('*** No outstanding orders.')
            return pd.DataFrame()
        #df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df['symbol,type,side,status,orderId,price,origQty,executedQty,cummulativeQuoteQty,updateTime'.split(',')]
        df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
        df = df.sort_values('updateTime', ascending=False)
        print('--[ orders ]\n',tabulate(df,headers="keys"))
        fn = os.getenv("USER_HOME","/Users/junma")
        fn += '/tmp/binance_open_orders.csv'
        df.to_csv( fn, index=0)
        print('-- saved:', fn)
        return df  
    
    @staticmethod
    def analyze_open_orders_cached() ->pd.DateOffset:
        fn = os.getenv("USER_HOME","/Users/junma")
        fn += '/tmp/binance_open_orders.csv'
        df = pd.read_csv( fn )
        df['datetime'] = df['datetime'].apply(pd.Timestamp)
        print(f'-- [trades from cached file: {fn}]')
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
        if df.empty: 
            print('*** No outstanding orders.')
            return pd.DataFrame()
        #df['dt'] = (tnow - df.updateTime.apply(int))/1000
        df = df['id,symbol,qty,price,commission,commissionAsset,isMaker,isBuyer,time'.split(',')]
        df['side'] = df.isBuyer.apply(lambda v: 'SELL' if not v else "BUY")
        df = df['id,symbol,side,qty,price,commission,commissionAsset,isMaker,time'.split(',')]
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
# test
def main_(ex, cbuy,csell,price,qty,sellbest,buybest):
    from butil.butils import get_binance_spot
    bid,ask = get_binance_spot(ex.ric.replace('-','/'))
    bid = float(bid)
    ask = float(ask)
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

import click
@click.command()
@click.option('--ric')
@click.option('--cbuy', is_flag=True,  default=False)
@click.option('--csell', is_flag=True,  default=False)
@click.option('--cancel', default='', help='comma-separated order ids to be canceled')
@click.option('--price',default=0.)
@click.option('--qty',default=0.)
@click.option('--sellbest', is_flag=True, default=False,help='judge from ask price, automatic create an order close to ask price')
@click.option('--buybest',  is_flag=True, default=False,help='judge from bid price, automatic create an order close to ask price')
def main(ric, cbuy,csell,cancel,price,qty,sellbest,buybest):
    from bbroker.settings import spot_ex
    assert 'USDT' in ric, r'Unsuported: {ric}'
    assert '-' in ric or '/' in ric, r'Unsupported: {ric}, use "-" or "/" in ric name'
    ex = BianceSpot(ric.replace('-','/'), spot_ex=spot_ex)
    
    if cancel:
        for oid in cancel.split(','):
            ex.cancel_order( oid )
    elif sellbest or buybest:
        assert qty>0, f"qty is required"
        main_(ex,False,False,0.,qty,sellbest,buybest)
    else:
        price = float(price)
        qty = float(qty)
        main_(ex,cbuy,csell,price,qty,sellbest,buybest)

if __name__ == '__main__':
    main()