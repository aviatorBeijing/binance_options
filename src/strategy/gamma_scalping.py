from functools import partial 

from butil.butils import get_maturity,get_binance_spot
from ws_bcontract import sync_fetch_ticker

"""
Ref: Trading Options Greeks, by Dan Passarelli, pg. 256
"""
class Asset:
    def __init__(self, ric, entry_price, quantity) -> None:
        self.ric = ric
        self.entry_price = entry_price
        self.quantity = quantity
    
    @staticmethod
    def get_spot_price(ric, ask=True):
        bid,ask = get_binance_spot( ric )
        if ask: return float(ask)
        else:
            return (float(bid)+float(ask))*.5 
    @staticmethod
    def get_options_price(contract):
        contract_price = None 
        def price_setter(c, data):
            c = data
        sync_fetch_ticker( contract, handler=partial( price_setter, (contract_price,) ) )
        return contract_price

    @staticmethod
    def get_options_greeks(ric):
        return  

    def value(self): raise Exception("Need impl.") 

class Spot(Asset):
    def __init__(self, ric, entry_price, quantity) -> None:
        super().__init__(ric, entry_price, quantity)
    def value(self, current_price):
        dv = (current_price - self.entry_price) * self.quantity
        return dv
    @property
    def delta(self):
        return 1.*self.quantity

    def __str__(self):
        return f"Spot: {self.quantity} @ ${self.entry_price}"
    def __repr__(self) -> str:
        return self.__str__()

class EuropeanOption(Asset):
    def __init__(self, contract, entry_price, quantity,
                nominal) -> None:
        """
        @brief
        param nominal   (int): The multiplier of a single contract, 
                              i.e., how many spot asset is covered by one option contract.
        """
        super().__init__(contract, entry_price, quantity)
        fds = contract.split("-")
        self.contract = self.ric = contract
        self.strike = float(fds[2])
        self.expiry = fds[1]
        self.putcall = "call" if fds[3] == 'C' else "put"

        self.greeks = self.get_options_greeks(self.contract)
        self.underlying = self.get_underlying( )
        self.maturity = self.get_maturity( )
        self.nominal = nominal 
        self.init_spot = Asset.get_spot_price(self.underlying)
        
        # Position delta
    def init(self):
        self.pdelta = self.greeks['delta']*self.nominal*self.quantity
        t = 1 if self.putcall == 'call' else -1
        s = 1 if self.quantity >0 else -1
        self.pdelta *= (t*s)

    def __str__(self):
        return f'''
      -- {self.ric} --
    underlying: {self.underlying}
    strike: {self.strike}
    expiry: {self.expiry}
    maturity: {self.maturity:.4f}
    greeks: {self.greeks}
    '''
    def get_underlying(self):
        fds = self.contract.split('-')
        return f"{fds[0]}/USDT"
    def get_maturity(self):
        dt = get_maturity( self.contract )
        return dt
    def value(self):
        if self.putcall == 'call': # value at Expiry
            return max(0, self.get_spot_price(self.underlying) - self.strike ) 
        elif self.putcall == 'put':
            return max(0, self.strike - self.get_spot_price(self.underlying) )
        else:
            raise Exception(f"Type = {self.putcall} is unkown.")
    def calc_options_greeks(self):
        S = self.get_spot_price( self.ric )
        K = self.strike
        T = self.maturity
        r = 5/100
    @property 
    def position_delta(self):
        return self.pdelta
        
    def on_market_move(self, new_spot):
        dd = self.on_spot_change( self.init_spot, new_spot)
        self.pdelta += dd
        if abs(dd)>0:
            print(f'  -- spot ${self.init_spot} to ${new_spot}, {((new_spot-self.init_spot)/self.init_spot*100):.1f}%,  {"SELL" if dd>0 else "BUY" if dd<0 else "STAY"} {abs(dd)} spot')
            #print(f'    -- delta change: {"+" if dd>0 else ""}{dd}, option delta: {self.pdelta}; need to {"SELL" if dd>0 else "BUY" if dd<0 else "STAY"} {abs(dd)} spot')
        self.init_spot = new_spot
        return dd
    
    def on_spot_change(self, from_spot, to_spot):
        if from_spot == to_spot: return 0

        delta_chg = ( to_spot - from_spot ) * self.greeks['gamma'] *self.nominal *self.quantity
        delta_chg = round(delta_chg) # convert to whole number as stock shares FIXME what about BTC, DOGE, etc.?
        return delta_chg
        
    def on_greeks_change(self, new_greeks={}): # TODO if delta,gamma changes, also need dynamic hedging
        pass

if __name__ == '__main__':
    s = EuropeanOption('BTC-240313-71000-P',1500,0.01,1)
    print( Asset.get_options_price( s.contract ) )


    nc = 20 # numbmer of calls
    call_price = 19.5
    p0 = 40 # initial spot price when creating portfolio

    contract = f'XYZ-240309-{p0}-C'
    Asset.get_spot_price = lambda e: p0
    c = EuropeanOption(contract, call_price, nc, 100)
    c.greeks = {'delta': 0.5, 'gamma': 2.8/20, 'theta': -0.5/20}
    c.init_spot = p0
    c.init()

    print(c)
    print( '-- delta of options: ',c.position_delta )

    spots = [] # spot stack

    spot = Spot('XYZ/USDT', p0, -1_000)
    spot.get_spot_price = lambda e: p0
    spots+=[spot]
    print( '-- delta of spot:', spot.delta)
    print( '-- total delta:', spot.delta + c.position_delta )

    def eod():
        return c.greeks['theta'] * c.nominal * c.quantity

    def adjust_delta( new_price, spots=spots ):
        c.get_spot_price = lambda e: new_price #Test
        delta_shift = c.on_market_move(new_price)
        spot = Spot('XYZ/USDT', new_price, -delta_shift) # Short more if delta increased
        spot.get_spot_price = lambda e: new_price # Test
        spots+=[spot]

        shares = sum([d.delta for d in spots])
        netp = c.position_delta + shares
        assert (netp)==0, f"Not fully hedged. Net={netp}"
    
    eods = []
    
    adjust_delta(42)
    adjust_delta(40)
    eods += [eod()]

    adjust_delta(39.6)
    adjust_delta(40)
    eods += [eod()]

    adjust_delta(40.5)
    adjust_delta(41)
    adjust_delta(41.5)
    adjust_delta(42)
    eods += [eod()]

    adjust_delta(38)
    eods += [eod()]

    # Weekends
    eods += [eod()]
    eods += [eod()]

    adjust_delta(38.25)
    eods += [eod()]

    p1 = 38.25
    for spot in spots[1:]:print( spot, spot.value(p1))
    scaples = sum( [d.value( p1 ) for d in spots[1:]] ) # The first spot is to construct initial Option+Spot portfolio. 
    print( f"-- scaple profits (after reduced of theta decay): \n\t${(scaples + sum(eods)):.2f}" )
    print( f"-- spot position left: {sum([d.delta for d in spots])}")