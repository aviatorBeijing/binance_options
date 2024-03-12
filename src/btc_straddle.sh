#!/bin/bash

CRYPTO=BTC
STRIKE=72000
DT=240315

PUT=$STRIKE
CALL=$STRIKE

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=0.01 #--user_premium=24.5 
