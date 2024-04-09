#!/bin/bash
source setup.sh

$PYTHON sentiments/spot_vol.py --rics=btc
$PYTHON sentiments/spot_vol.py --rics=eth
$PYTHON sentiments/spot_vol.py --rics=doge
