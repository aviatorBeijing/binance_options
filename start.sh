#!/bin/bash

export PYTHONPATH=$BINANCE_OPTIONS_DIR:$PYTHONPATH
cd server
export PYTHONPATH=../src:$PYTHONPATH;${PYTHON} app.py
