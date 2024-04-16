import os,datetime 
import pandas as pd 
from decimal import Decimal 

from sqlalchemy import create_engine, text

from butil.bsql import table_exists

uname = os.getenv('PG_USERNAME', 'junma')
psw = os.getenv('PG_PASSWORD', 'nethorse')
engine = bn_spot_engnine = create_engine(f"postgresql://{uname}:{psw}@localhost:5432/bn_spot")
ticker_snippet_tbl = "ticker_snippet"

def ticker_snippet_tbl_exists( ):
    return table_exists(ticker_snippet_tbl, bn_spot_engnine)
def init_ticker_snippet_tbl(df):
    df.to_sql(ticker_snippet_tbl, bn_spot_engnine, index=False)
    with bn_spot_engnine.connect() as conn:
        stmt =  f'ALTER TABLE {ticker_snippet_tbl} ADD PRIMARY KEY (ric);'
        conn.execute( text(stmt))
        conn.commit()
def read_latest_ticker(ric):
    ric = ric.replace('/','-')
    with bn_spot_engnine.connect() as conn:
        stmt =  f"SELECT bid,ask,ts,timestamp FROM {ticker_snippet_tbl} WHERE ric='{ric}';"
        res = conn.execute(stmt)
        if res:
            r = res.fetchall()
            assert len(r)==1
            r = r[0]
            bid =float(r[0]);ask=float(r[1]);ts=int(r[2]);timestamp=r[3]
            return bid,ask,ts,timestamp 
    return -1.,-1.,0,'n/a'

def write_latest_ticker(ric,bid,ask,ts,timestamp):
    assert isinstance(timestamp,str)
    assert isinstance(ts,int)
    ric = ric.replace('/','-')
    with bn_spot_engnine.connect() as conn:
        stmt =  f"""
INSERT INTO {ticker_snippet_tbl} (ric,bid,ask,ts,timestamp) VALUES ('{ric}',{bid},{ask},{ts},'{timestamp}') ON CONFLICT (ric) 
DO UPDATE SET bid={bid},ask={ask},ts={int(ts)},timestamp='{timestamp}';
"""
        conn.execute(text(stmt))
        conn.commit()