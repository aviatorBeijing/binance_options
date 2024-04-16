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