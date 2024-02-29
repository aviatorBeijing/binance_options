#!/bin/bash

CRYPTO=ETH
DT=240302
PUT=3400
CALL=3425

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
