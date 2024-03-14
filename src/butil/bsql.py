import os
import pandas as pd
from sqlalchemy import create_engine, text

uname = os.getenv('PG_USERNAME', 'junma')
psw = os.getenv('PG_PASSWORD', 'nethorse')
engine = bn_mkt_engine = create_engine(f"postgresql://{uname}:{psw}@localhost:5432/bn_options_mkt")
bidask_greeks_tbl = "bn_bidask_greeks"

def table_exists( tbname, _engine=None ):
    stmt = f'''
SELECT EXISTS 
(
        SELECT 1
        FROM information_schema.tables 
        WHERE table_name = '{tbname}'
);
    '''
    df = pd.read_sql(stmt, _engine if _engine else engine)
    tbname_exists = df.iloc[0].exists

    if tbname_exists: # Not only table exists, but also not empty
        stmt = f'SELECT FROM "{tbname}"'
        with _engine.connect() as conn:
            rst = conn.execute( text(stmt ) )
            rows = rst.fetchall()   
            if len(rows)>0:
                return True 
    return False

def bidask_table_exists():
    return table_exists(bidask_greeks_tbl, _engine=bn_mkt_engine)

def init_bidask_tbl(df):
    df.to_sql(bidask_greeks_tbl, bn_mkt_engine, index=False)
    with bn_mkt_engine.connect() as conn:
        stmt =  f'ALTER TABLE {bidask_greeks_tbl} ADD PRIMARY KEY (contract);'
        conn.execute( text(stmt))
        conn.commit()
def update_bidask_tbl( data:dict ):
    keys = ','.join( data.keys())
    vals = []
    kv_pairs = []
    for k in keys.split(","):
        if k != 'contract':
            vals += [ data[k] ]
            kv_pairs += [f'{k}={data[k]}']
        else:
            vals += [ f"'{data[k]}'" ]
    stmt = f'''
INSERT INTO {bidask_greeks_tbl} ({keys}) VALUES ({','.join([f"{v}" for v in vals])}) ON CONFLICT (contract)
DO UPDATE SET {','.join(kv_pairs)};
''' 
    try:
        with bn_mkt_engine.connect() as conn:
            conn.execute(text(stmt))
            conn.commit()
    except Exception as e:
        print(stmt)
        print(str(e))
        raise e
def fetch_bidask(contract):
    cols=f"""
SELECT * FROM {bidask_greeks_tbl} LIMIT 0;
"""
    stmt=f"""
SELECT * FROM {bidask_greeks_tbl} WHERE contract='{contract.upper()}';
"""
    with bn_mkt_engine.connect() as conn:
        conn.execute(text(cols)).fetchall()
        colnames = [desc[0] for desc in conn.cursor().description]
        print( colnames )

        recs = conn.execute( text(stmt)).fetchall()
        #if recs:
        #    dict( zip(self.DB_COLS, recs[0]))