import datetime 

"""
Ref: Trading Options Greeks, by Dan Passarelli, pg. 256
"""
class Asset:
    def __init__(self, ric, entry_price, quantity, is_long=True) -> None:
        self.ric = ric
        self.is_long = is_long 
        self.entry_price = entry_price
        self.quantity = quantity
    
    @classmethod
    def get_spot_price(self,ric):
        return 40
    @classmethod
    def get_options_price(self,ric):
        return 
    @classmethod
    def get_options_greeks(self,ric):
        return  {}
    def value(self): raise Exception("Need impl.") 

class Spot(Asset):
    def __init__(self, ric, entry_price, quantity, is_long=True) -> None:
        super().__init__(ric, entry_price, quantity, is_long)
        self.quantity = abs(self.quantity)
    def value(self):
        spot = self.get_spot_price(self.ric)
        dv = ( spot-self.entry_price ) * self.quantity
        return dv if self.is_long else -dv
    @property
    def delta(self):
        return 1.*self.quantity if self.is_long else -1.*self.quantity

class EuropeanOption(Asset):
    def __init__(self, ric, entry_price, quantity,
                nominal,
                is_long=True) -> None:
        """
        @brief
        param nominal   (int): The multiplier of a single contract, 
                              i.e., how many spot asset is covered by one option contract.
        """
        super().__init__(ric, entry_price, quantity, is_long)
        fds = ric.split("-")
        self.strike = float(fds[2])
        self.expiry = fds[1]
        self.putcall = "call" if fds[3] == 'C' else "put"

        self.greeks = self.get_options_greeks(self.ric)
        self.underlying = self.get_underlying( )
        self.maturity = self.get_maturity( )
        self.nominal = nominal 
        self.init_spot = self.get_spot_price(self.underlying)
        
        # Position delta
    def init(self):
        self.pdelta = self.greeks['delta']*self.nominal*self.quantity
        self.pdelta = self.pdelta if self.is_long else -self.pdelta

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
        fds = self.ric.split('-')
        return f"{fds[0]}/USDT"
    def get_maturity(self):
        fds = self.ric.split('-')
        ts = datetime.datetime.strptime('20'+fds[1], '%Y%m%d')
        tnow = datetime.datetime.utcnow()
        dt = (ts-tnow).total_seconds()/3600./24
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
        
    def on_new_spot(self, new_spot):
        delta_chg = ( new_spot - self.init_spot ) * self.greeks['gamma'] *self.nominal *self.quantity
        delta_chg = round(delta_chg) # convert to whole number as stock shares FIXME what about BTC, DOGE, etc.?
        self.pdelta += delta_chg
        self.init_spot = new_spot
        return delta_chg
    def on_greeks_change(self): # TODO if delta,gamma changes, also need dynamic hedging
        pass

if __name__ == '__main__':
    contract = 'XYZ-240309-40-C'
    nc = 20 # numbmer of calls
    call_price = 19.5

    c = EuropeanOption(contract, call_price, nc, 100, is_long=True)
    c.greeks = {'delta': 0.5, 'gamma': 2.8/20, 'theta': -0.5/20} # per contract
    c.init()

    print(c)
    print( '-- delta of options: ',c.position_delta )

    spots = [] # spot stack

    p0 = 40 # initial spot price when creating portfolio
    spot = Spot('XYZ/USDT', p0, 1_000, is_long=False);spots+=[spot]
    print( '-- delta of spot:', spot.delta)
    print( '-- total delta:', spot.delta + c.position_delta )

    # Day 1
    c.get_spot_price = lambda e: 42
    delta_shift = c.on_new_spot(42)
    print( '-- (day1.1) options delta  (shares):', delta_shift)
    spot = Spot('XYZ/USDT', 42, delta_shift, is_long=delta_shift<0);spots+=[spot]
    print( '-- (day1.1) 2nd spot delta:', spot.delta )
    shares = sum([d.delta for d in spots])
    print('-- (day1.1) spot positions:', shares)
    print( '-- (day1.1) total delta:', c.position_delta + shares)

    c.get_spot_price = lambda e: 40
    delta_shift = c.on_new_spot(40)
    print( '-- (day1.2) options delta  (shares):', delta_shift)
    spot = Spot('XYZ/USDT', 40, delta_shift, is_long=delta_shift<0);spots+=[spot]
    print( '-- (day1.2) new spot delta:', spot.delta )
    shares = sum([d.delta for d in spots])
    print('-- (day1.2) spot positions:', shares)
    print( '-- (day1.2) total delta:', c.position_delta + shares)
    g1 = gain_on_spot = sum( [d.value() for d in spots] )
    print('-- spot value change:', gain_on_spot)
    option_decay = c.greeks['theta'] * c.nominal * c.quantity
    print( '-- option value decay:', option_decay)
    print('-- net gain on 1st day:', gain_on_spot + option_decay )

    
    print('-- 2nd day')
    c.get_spot_price = lambda e: 39.6
    delta_shift = c.on_new_spot(39.6)
    print('  -- (day2) option delta (shares):', delta_shift)
    spot = Spot('XYZ/USDT', 39.6, delta_shift, is_long=delta_shift<0);spots+=[spot]
    print(f"  -- (day2) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    c.get_spot_price = lambda e: 40
    delta_shift = c.on_new_spot(40)
    spot = Spot('XYZ/USDT', 40, delta_shift, is_long=delta_shift<0);spots+=[spot]
    print(f"  -- (day2) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    shares = sum([d.delta for d in spots]) 
    print( '  -- (day2) EOD position delta:', c.position_delta + shares  )
    g2 = gain_on_spot = sum( [d.value() for d in spots] ) - g1
    option_decay = c.greeks['theta'] * c.nominal * c.quantity
    print('  -- net gain on 2nd day:', gain_on_spot + option_decay)

    """
    print('-- 3rd day')
    c.get_spot_price = lambda e: 40.5
    delta_shift = c.on_new_spot(40.5)
    print('  -- (day3) option delta (shares):', delta_shift)
    spot6 = Spot('XYZ/USDT', 40.5, delta_shift, is_long=delta_shift<0)
    print(f"  -- (day3) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    c.get_spot_price = lambda e: 41
    delta_shift = c.on_new_spot(41)
    spot7 = Spot('XYZ/USDT', 41, delta_shift, is_long=delta_shift<0)
    print(f"  -- (day3) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    c.get_spot_price = lambda e: 41.5
    delta_shift = c.on_new_spot(41.5)
    spot8 = Spot('XYZ/USDT', 41.5, delta_shift, is_long=delta_shift<0)
    print(f"  -- (day3) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    c.get_spot_price = lambda e: 42
    delta_shift = c.on_new_spot(42)
    spot9 = Spot('XYZ/USDT', 42, delta_shift, is_long=delta_shift<0)
    print(f"  -- (day3) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    shares = spot.delta + spot2.delta + spot3.delta + spot4.delta + spot5.delta \
        +spot6.delta +spot7.delta+spot8.delta+spot9.delta
    print( '  -- (day3) EOD position delta:', c.position_delta + shares, c.position_delta, shares  )

    print('-- 4th day')
    c.get_spot_price = lambda e: 38
    delta_shift = c.on_new_spot(38)
    print('  -- (day4) option delta (shares):', delta_shift)
    spot10 = Spot('XYZ/USDT', 38, delta_shift, is_long=delta_shift<0)
    print(f"  -- (day4) {'sell' if delta_shift>0 else 'long'} {abs(delta_shift)} shares")
    shares = spot.delta + spot2.delta + spot3.delta + spot4.delta + spot5.delta \
        +spot6.delta +spot7.delta+spot8.delta+spot9.delta\
            +spot10.delta
    print( '  -- (day4) EOD position delta:', c.position_delta + shares, c.position_delta, shares  )

    """