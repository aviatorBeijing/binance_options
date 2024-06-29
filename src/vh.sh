#!/bin/bash

# The "new_struct" option triggers the adoption of newly designed "unified" signal generator data structure.
# Without the option, the results fall back to old stucture, which is incompatible with the "unified" struct.

SYMS=$1

VOLT=68
if [[ $SERVER_LOCATION == 'local' ]];then
$PYTHON signals/reversal_from_volume_hikes.py --volt $VOLT --offline --new_struct --syms $SYMS
else
$PYTHON signals/reversal_from_volume_hikes.py --volt $VOLT --new_struct --syms $SYMS
fi
