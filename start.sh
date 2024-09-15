#!/bin/bash

export PYTHONPATH=$BINANCE_OPTIONS_DIR:$PYTHONPATH
cd server; make run
