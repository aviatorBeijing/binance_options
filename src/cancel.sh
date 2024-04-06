#!/bin/bash

CID=$1
CONTRACT=$2

$PYTHON bbroker/order_mgr.py --contract $CONTRACT --cancel_order_id=$CID
