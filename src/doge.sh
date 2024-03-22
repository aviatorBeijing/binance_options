#!/bin/bash

CRYPTO=DOGE
DT=240329
PUT=0.165
CALL=0.165

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
