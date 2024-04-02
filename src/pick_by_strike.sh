#!/bin/bash

K=$1
DT=$2
$PYTHON sentiments/pick_by_strike.py  --underlying=btc --strike=$K  --date4=$DT
