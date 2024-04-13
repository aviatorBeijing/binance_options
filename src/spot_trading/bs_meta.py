import pandas as pd 
from tabulate import tabulate
import datetime,os
class BianceSpot:
    def __init__(self,ric, spot_ex=None) -> None:
        self.ric = ric 
        self.ex = spot_ex
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