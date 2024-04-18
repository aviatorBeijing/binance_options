#!/bin/bash

CONTS=BTC-240419-60000-C,BTC-240419-60000-P
PCES=60000,63000

$PYTHON strategy/options_price_projection_from_spot_price.py --contracts $CONTS --projected_spot_prices $PCES
