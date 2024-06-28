#!/bin/bash

# The "new_struct" option triggers the adoption of newly designed "unified" signal generator data structure.
# Without the option, the results fall back to old stucture, which is incompatible with the "unified" struct.

SYMS=$1
$PYTHON signals/reversal_from_volume_hikes.py --volt 68 --offline --new_struct --syms $SYMS

#signals/reversal_from_volume_hikes.py --volt 68 --offline --syms doge,btc --new_struct
