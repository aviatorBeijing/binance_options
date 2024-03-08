#!/bin/bash

CRYPTO=BTC
PCE=66250
DT=240310

PUT=$PCE
CALL=$PCE

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=0.01 --user_premium=24.5 
