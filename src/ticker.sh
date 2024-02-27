#!/bin/bash

RICS=$1
python ws_bcontract.py --channel=ticker --rics $RICS
