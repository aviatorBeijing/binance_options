#!/bin/bash

SYMS=$1

INC=2
$PYTHON signals/climb_and_fall.py --syms $SYMS --offline 
#--up_inc=$INC --down_inc=$INC
