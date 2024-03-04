#!/bin/bash

CRYPTO=DOGE
DT=240308
PUT=0.169
CALL=0.169

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
