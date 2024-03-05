#!/bin/bash

CONTRACTS=$1
$PYTHON sentiments/vol_surf.py --contracts $CONTRACTS
