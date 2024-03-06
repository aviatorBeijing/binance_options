#!/bin/bash

DT=240306
LEFT=BTC-$DT-65000-P
RIGHT=BTC-$DT-65000-C
$PYTHON strategy/straddle.py --left $LEFT --right $RIGHT --size=0.1
