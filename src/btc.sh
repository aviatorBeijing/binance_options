#!/bin/bash

CRYPTO=BTC
PCE=67250
DT=240311

PUT=$PCE
CALL=$PCE

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
