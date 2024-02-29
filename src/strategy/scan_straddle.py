import datetime,os 
import click 
import pandas as pd

from strategy.straddle import _main as calc_straddle 

@click.command()
@click.option('--contracts')
@click.option('--sz', default=1.)
def main( contracts,sz ):
    contracts = contracts.split(',')
    puts = list(filter(lambda s: s.endswith('-P'), contracts))
    calls = list(filter(lambda s: s.endswith('-C'), contracts))

    recs = []
    for p in puts:
        for c in calls:
            resp = calc_straddle(p,c,vol=sz)
            recs += [resp]
    df = pd.DataFrame.from_records( recs )
    print( df )

if __name__ == '__main__':
    main()