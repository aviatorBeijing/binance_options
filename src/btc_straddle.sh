#!/bin/bash

# can also use "./atms.sh" to find all avalialbe ATM options, and select two ATM options.

CRYPTO=BTC
DT=240413

PUT=70500
CALL=70500

echo $PYTHON ws_bcontract.py --channel ticker --rics $CRYPTO-$DT-$PUT-P,$CRYPTO-$DT-$CALL-C
# $PYTHON ws_bcontract.py --channel ticker --rics $CRYPTO-$DT-$PUT-P,$CRYPTO-$DT-$CALL-C

$PYTHON strategy/straddle.py --left $CRYPTO-$DT-$PUT-P --right $CRYPTO-$DT-$CALL-C --size=0.01 --user_premium=23.45 #--check_parity 
# --user_premium=24.55 
