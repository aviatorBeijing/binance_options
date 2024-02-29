#!/bin/bash

$PYTHON -c "from butil.butils import get_bmex_next_funding_rate;print(   '-- bmex:   ', get_bmex_next_funding_rate('XBTUSD'))"
$PYTHON -c "from butil.butils import get_binance_next_funding_rate;print('-- binance:',get_binance_next_funding_rate('BTCUSDT'))"

$PYTHON -c "from butil.butils import get_bmex_next_funding_rate;print(   '-- bmex:   ', get_bmex_next_funding_rate('ETHUSD'))"
$PYTHON -c "from butil.butils import get_binance_next_funding_rate;print('-- binance:',get_binance_next_funding_rate('ETHUSDT'))"

