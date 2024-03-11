#!/bin/bash

source setup.sh

DT=240313
STRIKE=71000

CS=BTC-$DT-$STRIKE-C,BTC-$DT-$STRIKE-P
$PYTHON strategy/price_disparity.py --contracts $CS
