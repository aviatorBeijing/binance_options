#!/bin/bash

CRYPTO=ETH
DT=240306
PUT=3475
CALL=3475

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
