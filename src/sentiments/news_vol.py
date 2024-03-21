import datetime,os,click
import pandas as pd 
import matplotlib.pyplot as plt
from arch import arch_model

events = [
    ('2024-03-20 13:34', '美联储FOMC利率会议')
]

def get_data(ric):
    fdir = os.getenv('USER_HOME','') + f'/data/{ric.upper()}1H.csv'
    df = pd.read_csv( fdir )
    df['Timestamp'] = df['Timestamp'].apply(int).apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
    df.set_index('Timestamp', inplace=True,drop=True)
    return df

@click.command()
@click.option('--ric')
def main(ric):
    df = get_data(ric)
    print(df)

    close = df.Close.resample('1d').agg('last')
    rtn = close.dropna().pct_change().dropna()
    model = arch_model(rtn, mean='Zero', vol='GARCH', p=15,q=15)
    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=10)
    s  = rtn.head(10).std();print( s*s )
    print( yhat.variance.values[-1,:])

if __name__ ==  '__main__':
    main()