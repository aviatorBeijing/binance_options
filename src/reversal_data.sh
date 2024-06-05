#!/bin/bash
SERVER=3.114.152.67
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/reversal_data.tar.gz ./ && mv reversal_data.tar.gz ~/tmp
foodir=`pwd`
cd ~/tmp
tar xvfz reversal_data.tar.gz
cd $foodir
