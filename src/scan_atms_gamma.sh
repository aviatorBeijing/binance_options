#!/bin/bash

echo "Reading ~/tmp/_atms_btc.csv  (generated by ./atms.sh btc)"
$PYTHON strategy/scan_gamma.py --contracts `cat /home/ubuntu/tmp/_atms_btc.csv`
