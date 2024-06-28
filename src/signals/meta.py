
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
    