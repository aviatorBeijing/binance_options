#!/bin/bash

source setup.sh
CS=BTC-240313-66000-C,BTC-240313-66000-P
$PYTHON strategy/price_disparity.py --contracts $CS
