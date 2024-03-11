#!/bin/bash

CRYPTO=BTC
PCE=66000
DT=240313

PUT=$PCE
CALL=$PCE

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=0.01 #--user_premium=24.5 
