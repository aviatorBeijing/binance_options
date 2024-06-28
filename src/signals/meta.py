
import pandas as pd 
import enum 
class ActionT(enum.Enum):
    BUY = 'buy'
    SELL = 'sell'
    TP  = 'tp'
    SL  = 'sl'
class SignalEmitter(enum.Enum):
    CLIMB_AND_FALL  = 'climb_n_fall'
    VOLUME_HIKES    = 'volume_hikes'

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
            'emitter': [self.emitter.value],
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
        s = f' {self.emitter.value:12s} {self.ts}: {self.ric} {self.act}, ${self.price}, {self.sz}, {self.sz_f:.3f}'
        return s

def struct_last_trade_action(actions:list ):
    """
    Notice: the protocol of constructing the string matters.
    """
    if len(actions)>0:
        act = actions[-1]
        return f'{act.act.value},{act.ts},{act.price}' if len(actions)>0 else ""
    return ""

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
        actions:list, #': f'{last_action.act.value},{last_action.ts},{last_action.price}' if len(actions)>0 else "",
        price_now:float):
    return {
            'symbol': symbol,
            'emitter': actions[0].emitter.value,
            'end': end,
            'yrs': yrs,
            'single_max_gain_pct': single_max_gain_pct,
            'single_max_loss_pct': single_max_loss_pct,
            'cagr_pct': cagr_pct,
            'bh_cagr_pct': bh_cagr_pct,
            'sortino': sot,
            'bn_sortino': bh_sot,
            'maxdd': maxdd,
            'bh_maxdd': bh_maxdd,
            'last_action': struct_last_trade_action(actions),
            'price_now': price_now,
        }
