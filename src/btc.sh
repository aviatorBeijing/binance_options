#!/bin/bash

DT=240306
LEFT=BTC-$DT-64500-P
RIGHT=BTC-$DT-65500-C
$PYTHON strategy/straddle.py --left $LEFT --right $RIGHT --size=0.1
