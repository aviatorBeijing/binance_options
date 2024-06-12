#!/bin/bash
export PYTHONPATH=..:$PYTHONPATH
export PYTHONPATH=$FINAPI:$PYTHONPATH
export SIM_PLOT_ON=1
DTYPE=$DTYPE
echo "***********************"
echo "     DTYPE=$DTYPE"
echo "***********************"

SELL_STRATEGY=$4
RICS=$3
STRATEGY=$2
CAPITAL=$1

if [ -z "$SELL_STRATEGY" ];then
	echo "-- not provided, using SELL_STRATEGY=sell_one"
        SELL_STRATEGY=sell_one
else
	echo "-- sell strategy: $SELL_STRATEGY"
fi

#SELL_STRATEGY=sell_all
echo "  Sell strategy: $SELL_STRATEGY"

$PYTHON $FINAPI/api/intraday_trading.py --is_binance --dtype=$DTYPE --allocated_capital=$CAPITAL --skip_roe_filter --skip_cagr_filter --ric_sets=$RICS --strategy=$STRATEGY --sell_strategy=$SELL_STRATEGY


