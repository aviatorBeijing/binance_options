#!/bin/bash

SYMS=$1  #example: doge,btc

INC=2

if [[ $SERVER_LOCATION == 'local' ]];then
$PYTHON signals/climb_and_fall.py --syms $SYMS --offline 
else
$PYTHON signals/climb_and_fall.py --syms $SYMS
fi
#--up_inc=$INC --down_inc=$INC
