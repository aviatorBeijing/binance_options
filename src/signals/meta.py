
from ast import Num
import os
import pandas as pd 
import enum 
from sqlalchemy import text

from butil.bsql import emmiter_engine,table_exists
from butil.yahoo_api import get_asset_class
tbname = 'signals'


class ActionT(enum.Enum):
    BUY = 'buy'
    SELL = 'sell'
    TP  = 'tp'
    SL  = 'sl'

class SignalEmitter(enum.Enum):
    CLIMB_AND_FALL  = 'climb_n_fall'
    VOLUME_HIKES    = 'volume_hikes'
    EXT_MIXED       = 'ext_mixed'
    EXT_RSI         = 'ext_rsi'
    EXT_SENTIMENT   = 'ext_sentiment'

class Emitter: # Signal emmiter, i.e., different trading strategy
    def __init__(self,cap,span) -> None:
        self.cap = cap
        self.span = span
    def desc(self):
        raise Exception('To be overwritten.')

# Three externals
class ExtMixedEmitter(Emitter):
    T = SignalEmitter.EXT_MIXED
    @staticmethod
    def name(): return ExtMixedEmitter.T.value 
    def __init__(self,cap,bot,span ) -> None:
        super().__init__(cap,span)
        self.bot = bot # bot contains algo config parameters
    def desc(self):
        return f"span={self.span},bot={self.bot},cap={self.cap}"
class ExtRsiEmitter(Emitter):
    T = SignalEmitter.EXT_RSI
    @staticmethod
    def name(): return ExtRsiEmitter.T.value 
    def __init__(self,cap,bot,span ) -> None:
        super().__init__(cap,span)
        self.bot = bot # bot contains algo config parameters
    def desc(self):
        return f"span={self.span},bot={self.bot},cap={self.cap}"
class ExtSentimentEmitter(Emitter):
    T = SignalEmitter.EXT_SENTIMENT
    @staticmethod
    def name(): return ExtSentimentEmitter.T.value 
    def __init__(self,cap,bot,span ) -> None:
        super().__init__(cap,span)
        self.bot = bot # bot contains algo config parameters
    def desc(self):
        return f"span={self.span},bot={self.bot},cap={self.cap}"

class ClimbNFallEmitter(Emitter):
    T = SignalEmitter.CLIMB_AND_FALL
    @staticmethod
    def name(): return ClimbNFallEmitter.T.value 
    def __init__(self, cap, span, up_inc,down_inc ) -> None:
        super().__init__(cap,span)
        self.up_inc = up_inc # climb up percentage threshold
        self.down_inc = down_inc
    def desc(self):
        return f"span={self.span},up={self.up_inc},down={self.down_inc},cap={self.cap}"

class VolumeHikesEmitter(Emitter):
    T = SignalEmitter.VOLUME_HIKES 
    @staticmethod
    def name(): return VolumeHikesEmitter.T.value 
    def __init__(self, cap, span, volt ) -> None:
        super().__init__(cap,span)
        self.volt = volt # threshold of volume hiking ranking
    def desc(self):
        return f"span={self.span},volt={self.volt},cap={self.cap}"

class TradeAction:
    def __init__(self, emitter: SignalEmitter, sym: str,act:ActionT, price:float, sz:float, sz_f: float, ts:str) -> None:
        assert sz_f<=1., 'assume no leverage'
        self.emitter = emitter
        self.act = act 
        self.price = price
        self.sz=sz
        self.ts = ts 
        self.sz_f = sz_f

        sym = sym.upper()
        self.ric = f'{sym}/USDT' if not 'USDT' in sym else sym
    def to_df(self):
        df = pd.DataFrame.from_dict({
            'emitter': [self.emitter.name()],
            'ric': [self.ric],
            'action': [ self.act.value ],
            'price': [self.price],
            'sz': [self.sz],
            'sz_f': [self.sz_f],
            'ts': [ str(self.ts) ],
        })
        return df
    def is_buy(self):
        return self.act == ActionT.BUY

    def __str__(self) -> str:
        s = f' {self.emitter.name():12s} {self.ts}: {self.ric} {self.act}, ${self.price}, {self.sz}, {self.sz_f:.3f}'
        return s

def struct_last_trade_action(action:TradeAction ):
    """
    Notice: the protocol of constructing the string matters.
    """
    act = action
    sym = act.ric.replace('/USDT','').replace('-USD','')
    if get_asset_class(sym) != 'crypto':
        act.price = round(float(act.price),2)
    return f'{act.act.value},{act.price}'

def construct_lastest_signal(symbol:str,
        end:str,
        yrs:float,
        single_max_gain_pct:float,
        single_max_loss_pct:float,
        cagr_pct:float,
        bh_cagr_pct:float,
        sot:float,
        bh_sot:float,
        maxdd: float,
        bh_maxdd: float,
        last_act:TradeAction, #': f'{last_action.act.value},{last_action.ts},{last_action.price}' if len(actions)>0 else "",
        price_now:float):
    rec = {
            'symbol': symbol.upper(),
            'signal_ts': str(last_act.ts).replace('+00:00',''),
            'last_action': struct_last_trade_action(last_act),
            'price_now': price_now,
            'cagr_pct': cagr_pct,
            'sortino': sot,
            'maxdd': maxdd,
            'single_max_gain_pct': single_max_gain_pct,
            'single_max_loss_pct': single_max_loss_pct,
            'bh_cagr_pct': bh_cagr_pct,
            'bh_sortino': bh_sot,
            'bh_maxdd': bh_maxdd,
            'end': str(end),
            'yrs': yrs,
            'desc': last_act.emitter.desc(),
            'emitter': last_act.emitter.name(),
        }
    if not table_exists(tbname, emmiter_engine):
        with emmiter_engine.connect() as conn:
            pd.DataFrame.from_records( [ rec ] ).to_sql(tbname,conn,index=0)
            conn.commit()
            conn.execute( text(f'''
            ALTER TABLE {tbname} ADD PRIMARY KEY (symbol,signal_ts,last_action,emitter);
            '''))
            conn.commit()
    else:
        with emmiter_engine.connect() as conn:
            cols = ','.join( [f'"{s}"' for s in rec.keys()] )
            _f = lambda v: v if not isinstance(v,str) else f"'{v}'"
            fdvals = [ f'"{k}"={_f(v)}' for k,v in rec.items() ];fdvals = ','.join(fdvals)
            stmt = f'''
            INSERT INTO {tbname} ({cols}) VALUES ({','.join([ f"'{s}'" if isinstance(s, str) else f"{s}" for s in rec.values()] )}) 
            ON CONFLICT  (symbol,signal_ts,last_action,emitter) DO UPDATE SET {fdvals};
            '''
            conn.execute( text(stmt))
            conn.commit()
        
    return rec


def trade_recs2df(recs):
    df = pd.DataFrame.from_records( recs )
    df.sort_values('last_action', ascending=False, inplace=True)
    df['x'] = df.last_action.apply(lambda s: s.split(',')[1])
    df.sort_values('x', ascending=False, inplace=True)
    df.drop('x', inplace=True, axis=1)
    return df