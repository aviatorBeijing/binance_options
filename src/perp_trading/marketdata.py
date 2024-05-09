
from bbroker.settings import perp_ex

def adhoc_ticker(symbol='BTC/USDT')->tuple:
    symbol = symbol.replace('-','/')
    #qts = perp_ex.fetch_ticker(symbol)
    qts = perp_ex.public_get_ticker_bookticker({'symbol': symbol.replace('/','').replace('-','').upper()})
    bid,ask = qts['bidPrice'],qts['askPrice'] # bidQty,askQty
    bid = float(bid)
    ask = float(ask)

    spread = (ask-bid)/(ask+bid)*2
    assert spread< 5./10_000, f'spread is too wide: {spread} (bid:{bid},ask:{ask})'

    return float(bid),float(ask)