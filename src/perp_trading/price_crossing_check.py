import pandas as pd,os,click

from crossing_check import calculate_price_cross_counts

@click.command()
@click.option('--sym', default='sol')
@click.option('--levels',default=10)
def main(sym,levels):
    df = pd.read_csv(os.getenv("USER_HOME",'') + f'/tmp/perp_{sym.lower()}usdt_15m.csv')
    df['timestamp'] = df['starttime'].apply(pd.Timestamp)

    result = calculate_price_cross_counts(df, w=3, use_weight=False, num_levels=levels)
    print(result)

if __name__ == '__main__':
    main()

