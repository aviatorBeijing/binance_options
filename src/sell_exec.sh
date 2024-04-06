#!/bin/bash

CONTRACT=$1
PCE=$2
QTY=$3

$PYTHON bbroker/order_mgr.py --contract $CONTRACT \
	--qty=$QTY --price=$PCE --action sell \
	--execute
