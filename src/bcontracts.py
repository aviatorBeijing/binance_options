import datetime, os
import pandas as pd
import requests 
import click

@click.command()
@click.option('--underlying', default="BTC")
@click.option('--price')
@click.option('--low', help="low price bound of strike")
@click.option('--high', help="high price bound of strike")
@click.option('--contract', help="call or put")
def main(underlying, price, low, high, contract):
    endpoint = 'https://eapi.binance.com/eapi/v1/exchangeInfo'
    
    # Adding a User-Agent header is now recommended for eapi requests
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(endpoint, headers=headers)
    
    if resp.status_code == 200:
        data = resp.json()
        ts = data['serverTime']
        print(f'-- server time: {pd.to_datetime(ts, unit="ms")}')
        
        rics = data['optionSymbols']
        df = pd.DataFrame.from_records(rics)
        
        # Robust conversion for expiry and strike
        df['expiryDate'] = pd.to_datetime(df['expiryDate'].astype(float), unit='ms')
        df['strikePrice'] = df['strikePrice'].astype(float)

        # Robust filter extraction (searches for the correct filter type)
        def get_tick_size(filters):
            for f in filters:
                if f.get('filterType') == 'PRICE_FILTER':
                    return f.get('tickSize')
            return None
        
        df['tickSize'] = df['filters'].apply(get_tick_size)

        # Clean up dataframe
        cols_to_drop = ['filters', 'contractId', 'unit', 'id']
        df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
        df = df.sort_values(['expiryDate', 'symbol', 'strikePrice'], ascending=True)

        # Dynamic Pathing (Works on macOS and Ubuntu)
        home = os.path.expanduser("~")
        tmp_dir = os.path.join(home, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fn = os.path.join(tmp_dir, 'options_symbols.csv')
        df.to_csv(fn, index=False)

        # Filtering logic
        rcs = df.copy()
        if underlying:
            rcs = rcs[rcs.symbol.str.contains(underlying.upper())]
        if price:
            rcs = rcs[rcs.symbol.str.contains(str(price))]
        if contract:
            # Matches -C or -P specifically
            suffix = f"-{contract[0].upper()}"
            rcs = rcs[rcs.symbol.str.endswith(suffix)]

        symbols = rcs.symbol.values 
        
        if low:
            symbols = [s for s in symbols if float(s.split('-')[2]) >= float(low)]
        if high:
            symbols = [s for s in symbols if float(s.split('-')[2]) <= float(high)]

        print(','.join(symbols))
        print(f'{len(symbols)} contracts')
        print('-- saved: ', fn)
    else:
        print(f"Error: API returned status {resp.status_code}")

if __name__ == '__main__':
    main()
