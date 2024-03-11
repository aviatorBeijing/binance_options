#!/bin/bash

CRYPTO=BNB
DT=240313
PUT=515
CALL=515

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=1 
