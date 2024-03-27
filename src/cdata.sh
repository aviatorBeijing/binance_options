#!/bin/bash

source setup.sh

RICS=$1
$PYTHON ws_bcontract.py --channel ticker --rics $RICS #BTC-240329-70000-P
