#!/bin/bash

CONTRACTS=`cat /home/ubuntu/tmp/_atms_btc.csv`
$PYTHON strategy/scan_straddle.py --contracts $CONTRACTS --data
