#!/bin/bash
$PYTHON --version

PRICE=$1 # 57000
CONTRACT=$2 # call or put
$PYTHON bcontracts.py --price=$PRICE --contract=$CONTRACT
