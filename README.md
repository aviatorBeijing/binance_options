# binance_options

```
python -m pip install -r requirements.txt
```

Example:

```
export USER_HOME=<set_a_tmp_directory_for_data_cache>

cd src
python btc_straddle.py --left BTC-240308-50000-P --right BTC-240308-60000-C --size=1
```

Notes:
```
Make sure you are using OTM options, both call & put for straddle. Otherwise, meaningless.
Ref: https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-strangle
```
