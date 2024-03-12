#!/bin/bash

SRC=/home/ubuntu/src/binance_options/src
rm *.csv

source setup.sh

SYM=$1

$PYTHON $SRC/sentiments/atms.py --refresh_oi --underlying=$SYM
