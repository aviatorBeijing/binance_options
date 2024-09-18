import datetime,os,io
import click
from tqdm import tqdm
import pandas as pd
from contextlib import redirect_stdout 
        
from strategy.straddle import _main as calc_straddle 
from butil.butils import DEBUG 

def _main( contracts,sz ):
    contracts = contracts.split(',')
    puts = list(filter(lambda s: s.endswith('-P'), contracts))
    calls = list(filter(lambda s: s.endswith('-C'), contracts))
    puts = list(set(puts))
    calls = list(set(calls))

    recs = []
    tmps = []
    for p in puts:
        for c in calls:
            tmps += [(p,c)]
    for p,c in tqdm(tmps):
        if c.split('-')[1] != p.split('-')[1]: # Match the expiry
            continue 
        if DEBUG: print('--', p, c)
        with redirect_stdout(io.StringIO()) as f:
            resp, _ = calc_straddle(p,c,vol=sz)
        if resp:
            recs += [resp]
    df = pd.DataFrame.from_records( recs )
    
    try:
        df['x'] = df.be_returns.apply(lambda e: len(e))
    except Exception as e:
        print( df )
        raise e
    df = df[df.x>0]
    try:
        df['x'] = df.be_returns.apply(lambda e: e[-1])
        df = df.sort_values(['x'], ascending=True)
        #df.drop(['x'],inplace=True,axis=1)
    except Exception as e:
        df.to_csv(f'{os.getenv("USER_HOME")}/tmp/debug.csv')

    df['break_even_low'] = df.be_prices.apply(lambda e: e[0])
    df['break_even_high'] = df.be_prices.apply(lambda e: e[-1])
    df['spot_down_r'] = df.be_returns.apply(lambda e: f"{(e[0]*100):.1f}%")
    df['spot_up_r'] = df.be_returns.apply(lambda e: f"{(e[-1]*100):.1f}%")
    df['straddle_down_r'] = df.straddle_returns.apply(lambda e: f"{(e[0]*100):.1f}%")
    df['straddle_up_r'] = df.straddle_returns.apply(lambda e: f"{(e[-1]*100):.1f}%")
    return df 

@click.command()
@click.option('--contracts')
@click.option('--data', is_flag=True,default=False)
@click.option('--sz', default=1.)
def main(contracts, data, sz):
    if data:
        from ws_bcontract import _main as ws_connector
        ws_connector( contracts,'ticker')
    else:
        df = _main(contracts,sz)
        #print(df.columns)
        df.drop(['is_taker','be_prices','be_returns','paid_premium','straddle_returns'], inplace=True, axis=1)
        import tabulate
        print( tabulate.tabulate(df,headers="keys") )


if __name__ == '__main__':
    main()
