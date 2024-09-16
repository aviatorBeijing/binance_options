#!/bin/bash
cd $BINANCE_OPTIONS_DIR
$PYTHON sentiments/atms.py  --check_price_ranges --update --refresh_oi
