#!/bin/bash

CRYPTO=ETH
DT=240313
PUT=4025
CALL=4025

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
