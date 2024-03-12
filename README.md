# Options Trading Strategies (via Binance API)

```
python -m pip install -r requirements.txt

export USER_HOME=<set_a_tmp_directory_for_data_cache>
```

Example (Long straddle):

```
cd src
source setup.sh

python strategy/straddle.py --left BTC-240313-71000-P --right BTC-240313-71000-C --size=0.1
python strategy/straddle.py --left BNB-240301-385-P --right BNB-240301-400-C --size=1
```

Example (Gamma scalping):
```
cd src
source setup.sh
python strategy/gamma_scalping.py 
```

Debug mode:
```
export BINANCE_DEBUG=1
```

Notes:
```
Only support BTC,BNB, and DOGE contracts, for now.
Ref: https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-strangle
```
