#!/bin/bash

DT=240308
LEFT=BTC-$DT-62000-P
RIGHT=BTC-$DT-62000-C
$PYTHON strategy/straddle.py --left $LEFT --right $RIGHT --size=0.1
