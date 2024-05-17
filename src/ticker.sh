#!/bin/bash
$PYTHON --version
RICS=$1
$PYTHON ws_bcontract.py --channel=ticker --ric $RICS
