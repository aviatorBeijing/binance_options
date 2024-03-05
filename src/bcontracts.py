import datetime,os
import pandas as pd
import requests 
import click

@click.command()
@click.option('--underlying', default="BTC")
@click.option('--price')
@click.option('--low', help="low price bound of strike")
@click.option('--high', help="high price bound of strike")
@click.option('--contract', help="call or put")
def main(underlying,price,low,high,contract):
    endpoint='https://eapi.binance.com/eapi/v1/exchangeInfo'
    resp = requests.get(endpoint)
    if resp:
        resp  = resp.json()
        ts = resp['serverTime']
        print('-- server time:', pd.Timestamp( datetime.datetime.fromtimestamp( int(float(ts)/1000)) ) )
        rics = resp['optionSymbols']
        df = pd.DataFrame.from_records( rics )
        
        df.expiryDate = df.expiryDate.apply(float).apply(lambda v:v/1000).apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
        df.strikePrice = df.strikePrice.apply(float)
        df['tickSize'] = df['filters'].apply(lambda v: v[0]['tickSize'] )
        #df['minQty'] = df['filters'].apply(lambda v: v[1]['minQty'])

        df.drop(['filters','contractId','unit','id'], axis=1, inplace=True)
        df = df.sort_values(['expiryDate','symbol','strikePrice'], ascending=True)
        #print(df)

        fn = os.getenv('USER_HOME','/Users/junma')
        fn += '/tmp/options_symbols.csv'
        df.to_csv( fn , index=False)
        
        symbols = df[df.symbol.str.contains(underlying.upper())].symbol.values
        # print(','.join(symbols))

        print('-- saved: ', fn)

        rcs = df.copy()
        if underlying:
            rcs = rcs[ rcs.symbol.str.contains(underlying.upper()) ]
        if price:
            rcs = rcs[ rcs.symbol.str.contains(price) ]
        if contract:
            rcs = rcs[ rcs.symbol.str.endswith( f"-{contract[0].upper()}" ) ]
        rcs = rcs.symbol.values 
        if low:
            rcs = list(
                filter(
                    lambda c: float(c.split('-')[2]) >= float(low), rcs
                )
            )
        if high:
            rcs = list(
                filter(
                    lambda c: float(c.split('-')[2]) <= float(high), rcs
                )
            )

        print(','.join( rcs ))
        print(len(rcs), ' contracts')
        print('saved: ',fn)


if __name__ == '__main__':
    main()
