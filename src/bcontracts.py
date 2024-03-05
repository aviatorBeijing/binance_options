import datetime,os
import pandas as pd
import requests 
import click

@click.command()
@click.option('--price')
@click.option('--contract', help="call or put")
def main(price,contract):
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
        
        symbols = df[df.symbol.str.contains('BTC')].symbol.values
        # print(','.join(symbols))

        print('-- saved: ', fn)

        rcs = df.copy()
        if price:
            rcs = rcs[ rcs.symbol.str.contains(price) ]
        if contract:
            rcs = rcs[ rcs.symbol.str.endswith( f"-{contract[0].upper()}" ) ]
        rcs = rcs.symbol.values 
        print(','.join( rcs ))


if __name__ == '__main__':
    main()
