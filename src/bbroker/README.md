***Example scripts:***


```

python bbroker/option_lookup.py --date 260828 --target_price 75000

python bbroker/straddle_calc.py --call BTC-260828-75000-C --put BTC-260828-75000-P --size 0.01 --action buy --iv 35

python bbroker/ext_order_mgr.py order --action buy --contract BTC-260828-76000-C --qty 0.01 --limit --execute --order_price 500

python bbroker/ext_order_mgr.py order --action buy --contract BTC-260828-76000-C --qty 0.01 --chase
python bbroker/ext_order_mgr.py order --action buy --contract BTC-260828-76000-C --qty 0.01 --chase --execute

python bbroker/ext_order_mgr.py status

python bbroker/ext_order_mgr.py cancel --contract BTC-260828-76000-C --order_id 851385

python bbroker/ext_order_mgr.py close --contracts "BTC-260828-76000-C,BTC-260828-76000-P"
python bbroker/ext_order_mgr.py close --contracts "BTC-260828-76000-C,BTC-260828-76000-P" --execute

```

