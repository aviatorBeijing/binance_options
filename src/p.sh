#!/bin/bash
source /home/ubuntu/.bashrc
export PYTHONPATH=$pwd:$PYTHONPATH
/usr/bin/python3 spot_trading/portfolio.py
