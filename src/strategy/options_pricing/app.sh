#!/bin/bash

PCS=$2
CONTRACTS=$1

python options_price_projection_from_spot_price.py --contracts CONTRACTS --projected_spot_prices PCS
