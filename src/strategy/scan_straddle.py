import datetime,os 
import click 
import pandas as pd

from strategy.straddle import _main as calc_straddle 

def _main( contracts,sz ):
    contracts = contracts.split(',')
    puts = list(filter(lambda s: s.endswith('-P'), contracts))
    calls = list(filter(lambda s: s.endswith('-C'), contracts))

    recs = []
    for p in puts:
        for c in calls:
            resp = calc_straddle(p,c,vol=sz)
            recs += [resp]
    df = pd.DataFrame.from_records( recs )

    df['break_even_low'] = df.be_prices.apply(lambda e: e[0])
    df['break_even_high'] = df.be_prices.apply(lambda e: e[3])
    df['low_r'] = df.be_ratios.apply(lambda e: f"{(e[0]*100):.1f}%")
    df['high_r'] = df.be_ratios.apply(lambda e: f"{(e[3]*100):.1f}%")
    
    df.drop(['is_taker','be_prices','be_ratios'], inplace=True, axis=1)
    print( df )

@click.command()
@click.option('--contracts')
@click.option('--data', is_flag=True,default=False)
@click.option('--sz', default=1.)
def main(contracts, data, sz):
    if data:
        from ws_bcontract import _main as ws_connector
        ws_connector( contracts,'ticker')
    else:
        _main(contracts,sz)


if __name__ == '__main__':
    main()