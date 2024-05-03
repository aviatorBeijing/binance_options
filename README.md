# Trading Toolkits: Spot & Options (via Binance API)

```
python -m pip install -r requirements.txt

export USER_HOME=<set_a_tmp_directory_for_data_cache>
```

Example#1 (Long straddle):

```
cd src
source setup.sh

python strategy/straddle.py --left BTC-240313-71000-P --right BTC-240313-71000-C --size=0.1
python strategy/straddle.py --left BNB-240301-385-P --right BNB-240301-400-C --size=1
```

Example#2 (Gamma scalping):
```
cd src
source setup.sh
python strategy/gamma_scalping.py 
```

Debug mode:
```
export BINANCE_DEBUG=1
```

Example#3 (Spot/Forward grid trading):
```
(The beauty of the grid trading is in general, the choice of trading price at the moment is NOT sensitive. AVG(ohlc) is chosed in algo.)
cd src
python strategy/grid_trading.py --ric BTC/USDT --nominal=0.01 --stop_loss=0               (realtime data)
python strategy/grid_trading.py --ric BTC/USDT --nominal=0.01 --stop_loss=0 --test        (using cached data)
python strategy/grid_trading.py --ric BTC/USDT --nominal=0.01 --stop_loss=0 --random_sets (try a few subsets of data)

python strategy/grid_trading.py --ric BTC/USDT --nominal=0.01 --stop_loss=-0.2            (stop-loss -20%)
python strategy/grid_trading.py --ric BTC/USDT --nominal=0.01 --stop_loss=-0.2 --max_pos=100   (use max_pos to mimic sizing)
```

Example #$ (Term-structure of C/P pair, ignoring volatility changes, and interest rate)
```
python brisk/pricing.py --contracts BTC-240506-59000-C,BTC-240506-59000-P
```

Notes:
```
Only support BTC,BNB, and DOGE contracts, for now.
Ref: https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-strangle
```
