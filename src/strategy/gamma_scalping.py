import datetime
from functools import partial
from subprocess import call 

from butil.butils import get_maturity,get_binance_spot,get_underlying, DEBUG
from ws_bcontract import sync_fetch_ticker
import numpy  as np

"""
Ref: Trading Options Greeks, by Dan Passarelli, pg. 256
"""
class Asset:
    def __init__(self, ric, entry_price, quantity) -> None:
        self.ric = ric
        self.entry_price = entry_price
        self.quantity = quantity
    
    @staticmethod
    def get_spot_price(ric, ask=True) -> float:
        bid,ask = get_binance_spot( ric )
        if ask: return float(ask)
        else:
            return (float(bid)+float(ask))*.5 
    @staticmethod
    def get_options_price(contract):
        contract_price = sync_fetch_ticker( contract )
        """
        {"last_trade": "3105", "bid": "3015", "ask": "3050", "bidv": "2.97", "askv": "1.5", "delta": "0.48350074", "gamma": "0.00004636", "theta": "-208.4755263", "vega": "43.74505017", "impvol": "0.78341722", "impvol_bid": "0.77941656", "impvol_ask": "0.78741789"}
        """
        if contract_price:
            bid, ask = contract_price['bid'], contract_price['ask']
            last_trade = contract_price['last_trade']
            return float(bid), float(ask), float(last_trade)
        return None,None,None

    @staticmethod
    def get_options_greeks(contract):
        gks = sync_fetch_ticker( contract )
        if gks:
            delta, gamma, theta, vega = gks['delta'], gks['gamma'], gks['theta'], gks['vega']
            impvol, impvol_bid, impvol_ask = gks['impvol'], gks['impvol_bid'], gks['impvol_ask']
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega),
                'impvol': float(impvol),
                'impvol_bid': float( impvol_bid ),
                'impvol_ask': float( impvol_ask ),
            }
        return {}  

    def value(self): raise Exception("Need impl.") 

class Spot(Asset):
    def __init__(self, ric, entry_price, quantity) -> None:
        """
        @param quantity: if <0, indicates taking short position.
        """
        super().__init__(ric, entry_price, quantity)
    def value(self, current_price=None ): # dollar value
        if not current_price: # If not provided by caller
            current_price = Asset.get_spot_price(self.ric)
        dv = (current_price - self.entry_price) * self.quantity
        return dv
    @property
    def delta(self):
        return 1.*self.quantity

    def __str__(self):
        return f"Spot: {self.quantity:.6f} @ ${self.entry_price:.6f}"
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

        self.greeks = Asset.get_options_greeks(self.contract)
        self.underlying = get_underlying( contract)
        self.maturity = self.get_maturity()
        self.nominal = nominal 

        self.init_spot = Asset.get_spot_price(self.underlying)
        self.bid,self.ask,self.last_trade = Asset.get_options_price(contract)
        
        # Position delta
    def init(self):
        self.pdelta = self.normalized_delta()
        return self

    def normalized_delta(self):
        dt = self.greeks['delta']*self.nominal*self.quantity
        dt *= 1 if self.quantity >0 else -1
        return dt

    def update_greeks(self):
        self.greeks = Asset.get_options_greeks(self.contract)
        self.bid,self.ask,self.last_trade = Asset.get_options_price(self.contract)
        
    def __str__(self):
        return f'''
      -- {self.ric} --
    underlying: {self.underlying}
    nominal: {self.nominal}
    strike: {self.strike}
    expiry: {self.expiry}
    maturity: {self.maturity:.4f}
    greeks: {self.greeks}
    '''
    def get_maturity(self):
        dt = get_maturity( self.contract )
        return dt
    
    def intrinsic_value(self):
        if self.putcall == 'call': # value at Expiry
            v = max(0, Asset.get_spot_price(self.underlying) - self.strike ) 
        elif self.putcall == 'put':
            v = max(0, self.strike - Asset.get_spot_price(self.underlying) )
        else:
            raise Exception(f"Type = {self.putcall} is unkown.")
        return v 
    def time_value(self):
        return  -self.maturity * self.greeks['theta'] * self.nominal*self.quantity

    def value(self):
        return self.intrinsic_value() + self.time_value()

    def calc_bsm_greeks(self):
        S = Asset.get_spot_price( self.ric )
        K = self.strike
        T = self.maturity
        r = 5/100
    @property 
    def position_delta(self):
        return self.pdelta
        
    def on_market_move(self):
        self.update_greeks()
        
        new_spot = Asset.get_spot_price( self.underlying )
        chg = (new_spot-self.init_spot)/self.init_spot
        if abs(chg) < 1e-12: # No change
            return 0, None
        if abs(chg) < 50/1000:
            print(f'*** trivial: {self.init_spot} to {new_spot}, {(chg*100):.3f}%')
            return 0, None
        delta_change = self.on_spot_change( self.init_spot, new_spot) # delta chg from spot price change
        self.pdelta += delta_change
        addition = None
        ts = datetime.datetime.utcnow().timestamp();ts = int(ts)
        if abs(delta_change)>0:
            #print(f'  -- [{ts}] spot change: {(chg*100):.3f}%,  {"SELL" if delta_change>0 else "BUY" if delta_change<0 else "STAY"} {abs(delta_change):.6f} @ ${new_spot} {self.underlying}')
            addition = Spot(self.underlying, new_spot, -delta_change)
            
        self.init_spot = new_spot # Reset mark price after rebalnced
        return delta_change, addition
    
    def on_spot_change(self, from_spot, to_spot): # Gamma induced delta change
        if from_spot == to_spot: return 0
        delta_chg = ( to_spot - from_spot ) * self.greeks['gamma'] *self.nominal *self.quantity
        #delta_chg = round(delta_chg) # convert to whole number as stock shares FIXME what about BTC, DOGE, etc.?
        return delta_chg

if __name__ == '__main__':
    #s = EuropeanOption('BTC-240313-71000-P',1500,0.01,1)
    #print( Asset.get_options_price( s.contract ) )
    #print( s.greeks )


    nc = 20 # numbmer of calls
    call_price = 19.5
    p0 = 40 # initial spot price when creating portfolio

    sync_fetch_ticker = lambda c: {'delta': 0.5, 'gamma': 2.8/20, 'theta': -0.5/20,
    'bid': call_price, 'ask': call_price, 'last_trade': call_price,'vega': 0,
    'impvol':0, 'impvol_bid': 0, 'impvol_ask':0}

    contract = f'XYZ-240509-{p0}-C'
    Asset.get_spot_price = lambda e: p0
    c = EuropeanOption(contract, call_price, nc, 100)
    c.init_spot = p0
    c.init()

    print(c)
    print( '-- delta of options: ',c.position_delta )

    spots = [] # spot stack

    spot = Spot('XYZ/USDT', p0, -c.position_delta)
    Asset.get_spot_price = lambda e: p0
    spots+=[spot]
    print( '-- delta of spot:', spot.delta)
    print( '-- total delta:', spot.delta + c.position_delta )

    def eod():
        return c.greeks['theta'] * c.nominal * c.quantity

    def adjust_delta( new_price, spots=spots ):
        Asset.get_spot_price = lambda e: new_price #Test
        delta_shift, spot = c.on_market_move()
        if spot: # Threshold reached and a new position is necessary
            Asset.get_spot_price = lambda e: new_price # Test
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
    Asset.get_spot_price = lambda v: p1
    for spot in spots[1:]:
        v = spot.value()
        print( spot, f"{v:.6f}")
    scaples = sum( [d.value() for d in spots[1:]] ) # The first spot is to construct initial Option+Spot portfolio. 
    print( f"-- scaple profits (after reduced of theta decay): \n\t${(scaples + sum(eods)):.2f}" )
    print( f"-- spot position left: {sum([d.delta for d in spots])}")
    print( f"-- option contract value left: ${c.value():.2f}")