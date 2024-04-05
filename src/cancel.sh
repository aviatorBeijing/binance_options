#!/bin/bash

CONTRACT=$1
CID=$2

$PYTHON bbroker/order_mgr.py --contract $CONTRACT --cancel_order_id=$CID
