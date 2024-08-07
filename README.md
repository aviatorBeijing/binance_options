# Trading Spot, Perpetual & Options (via Binance API)

( Be aware there are some PSQL database settings needed. I'll update later.)
```
python -m pip install -r requirements.txt

export USER_HOME=<set_a_tmp_directory_for_data_cache>
```

Summaries [Options]:
| Command  | Shorthand | Function |
| ------------- | ------------- | ------------- |
| ```./buy.sh```  |   | place single options order |
| ```./cancel.sh```  |  |cancel an option order|
|```python bbroker/check_status.py```|```./status.sh```|check options account positions, pnl, etc.|
|```python brisk/pricing.py --contract BTC-240607-56500-C --user_cost 585```||check the pnl against spot price movement|
|```python strategy/straddle.py --left BTC-240807-56500-C --right BTC-240807-56500-P --size=0.1```||pnl monitor of trading a straddle|
|```python strategy/straddle.py --left BTC-240807-56500-C --right BTC-240807-56500-P --size=0.1 --user_premium 95.5```||pnl calc of a straddle position|
|``````|||

Summaries [Perp]  (DOGE/USDT):
| Command  | Shorthand | Function |
| ------------- | ------------- | ------------- |
|```./perp_trading/doge/check.sh```|```./pp```|check current perp positions & pnl|
|```./perp_trading/doge/buyup.sh```|```./pu```|("u" stands for "upward") send a order pair: buy at best ask, sell at 50bps above|
|```./perp_trading/doge/selldown.sh```|```./pd```|("d" stands for "downward") send a order pair: sell at best bid, buy at 50bps below|
|```./perp_trading/doge/cancel.sh```|```./pc```|cancel orders by order IDs, comma separated.|
|```./perp_trading/doge/instant_pair.sh```|```./ps```|send an order pair: at 25bps each side|
|``````|||

Summaries [Spot]:
| Command  | Shorthand | Function |
| ------------- | ------------- | ------------- |
|```python spot_trading/grid.py  --ric BTC-USDT --uniform_grid_gap 50```||grid generator, 50bps gap|
|``````|||

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

Example #4 (Term-structure of C/P pair, ignoring volatility changes, and interest rate)
```
python brisk/pricing.py --contracts BTC-240506-59000-C,BTC-240506-59000-P
```

Notes:
```
Only support BTC,BNB, and DOGE contracts, for now.
Ref: https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-strangle
```

## SPOT (MPT)
```
python signals/reversal_from_volume_hikes.py --syms btc,doge,sol --volt 68
```
Comparison of "Volume hiking strategy" and "Plain buy&hold strategy":
![Screenshot 2024-06-06 at 12 00 22](https://github.com/aviatorBeijing/binance_options/assets/5878463/09cf9a0f-d916-448f-9c76-1a60296fdbcb)
