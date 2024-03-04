#!/bin/bash

DT=240305
LEFT=BTC-$DT-65500-P
RIGHT=BTC-$DT-65500-C
$PYTHON strategy/straddle.py --left $LEFT --right $RIGHT --size=0.1
