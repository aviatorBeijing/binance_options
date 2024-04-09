import ccxt,os,datetime
from tabulate import tabulate
import pandas as pd 

apikey = os.getenv('BINANCE_MAIN_OPTIONS_APIKEY', None)
secret = os.getenv('BINANCE_MAIN_OPTIONS_SECRET', None)
ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'option',
    }
})
ex.load_markets()

spot_ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'spot',
    }
})
spot_ex.load_markets()

class BianceSpot:
    def __init__(self,ric) -> None:
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
        return df  
    def check_trades_today(self)  -> pd.DataFrame:
        dt = datetime.datetime.utcnow().timestamp() - 24*3600
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
        print('--[ trades in 24 hours ]\n',tabulate(df,headers="keys"))
        return df  