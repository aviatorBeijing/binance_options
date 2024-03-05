#!/bin/bash
$PYTHON --version

$PYTHON bcontracts.py --contract=call --underlying=btc --low 60000 --high 75000
$PYTHON bcontracts.py --contract=call --underlying=eth --low 3200 --high 4000
$PYTHON bcontracts.py --contract=put --underlying=doge --low 0.15 --high 0.25
