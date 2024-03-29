#!/bin/bash

CRYPTO=DOGE
DT=240405
PUT=0.217
CALL=0.217

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
