#!/bin/bash

cd $BINANCE_OPTIONS_DIR/strategy/gamma_scalping

$PYTHON merton_jump_model.py
open $USER_HOME/tmp/merton_jump_model.png
