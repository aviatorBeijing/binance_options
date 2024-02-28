# binance_options
## Toolbox to use Binance Options trading APIs 

```
python -m pip install -r requirements.txt
```

Example (Long straddle):

```
export USER_HOME=<set_a_tmp_directory_for_data_cache>

cd src
python strategy/straddle.py --left BTC-240301-56000-P --right BTC-240301-58000-C --size=0.1
python strategy/straddle.py --left BNB-240301-385-P --right BNB-240301-400-C --size=1
```

Debug mode:
```
export BINANCE_DEBUG=1
```

Notes:
```
Only support BTC,BNB, and DOGE contracts, for now.
Make sure you are using OTM options, both call & put for straddle. Otherwise, meaningless.
Ref: https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-strangle
```
