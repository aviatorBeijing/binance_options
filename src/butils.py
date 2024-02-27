import os 

DATADIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/options'
if not os.path.exists( DATADIR):
    os.makedirs( DATADIR )
print('-- data dir:', DATADIR)
