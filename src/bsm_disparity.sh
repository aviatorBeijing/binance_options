#!/bin/bash

source setup.sh
CS=BTC-240310-69750-C,BTC-240310-69750-P
$PYTHON strategy/price_disparity.py --contracts $CS
