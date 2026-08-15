import datetime
import os
import sys
import time
import click
import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

from butil.butils import binance_spot, get_binance_spot
from strategy.price_disparity import extract_specs

R_RATE = 0.02  # Constant risk-free interest rate (2%)


def fetch_oi(expiry, underlying='BTC'):
    try:
        recs = requests.get(
            url='https://eapi.binance.com/eapi/v1/openInterest',
            params={'underlyingAsset': underlying.upper(), 'expiration': expiry}
        ).json()
        df = pd.DataFrame.from_records(recs)
        return df
    except Exception as e:
        print("*** fetch_oi failed: ", expiry, underlying)
        return pd.DataFrame()


def fetch_contracts(underlying):
    endpoint = 'https://eapi.binance.com/eapi/v1/exchangeInfo'
    resp = requests.get(endpoint)
    if resp:
        resp = resp.json()
        ts = resp['serverTime']
        print('-- server time:', pd.Timestamp(datetime.datetime.fromtimestamp(int(float(ts) / 1000))))
        rics = resp['optionSymbols']
        df = pd.DataFrame.from_records(rics)

        df.expiryDate = df.expiryDate.apply(float).apply(lambda v: v / 1000).apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
        df.strikePrice = df.strikePrice.apply(float)
        df['tickSize'] = df['filters'].apply(lambda v: v[0]['tickSize'])

        df = df[df.symbol.str.startswith(underlying.upper())]
        df = df.sort_values(['expiryDate', 'symbol', 'strikePrice'], ascending=True)
        df.reset_index(inplace=True, drop=True)
        return df
    return pd.DataFrame()


def get_atm(underlying, df):
    if os.getenv("YAHOO_LOCAL", None):
        print("\n", "*" * 10, " faking prices on local environment", "\n")
        bid, ask = 59000, 59000
    else:
        bid, ask = binance_spot(f"{underlying.upper()}/USDT")
    
    # Ensure strikePrice numeric
    df['strikePrice'] = df['symbol'].apply(lambda s: float(s.split('-')[2]))
    df['distance'] = df.strikePrice - (bid + ask) * 0.5
    recs = {}
    for expiry in sorted(list(set(df.expiryDate.astype(str).values))):
        edf = df[df.expiryDate.astype(str) == expiry]
        s1 = edf[edf.distance >= 0].sort_values(['distance'], ascending=True).head(2).symbol.values
        s2 = edf[edf.distance < 0].sort_values(['distance'], ascending=False).head(2).symbol.values
        recs[expiry] = list(sorted(list(s1) + list(s2)))
    return recs


def _dir():
    fdir = os.getenv("USER_HOME", "") + '/tmp/binance_options/'
    fdir += datetime.datetime.strftime(datetime.datetime.today(), '%Y_%m_%d')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    return fdir


def refresh_contracts(underlying, update=False):
    fn = f"{_dir()}/_all_binance_contracts_{underlying.lower()}.csv"
    if update or not os.path.exists(fn):
        df = fetch_contracts(underlying)
        df['expiry'] = df.symbol.apply(lambda s: s.split('-')[1])
        df.to_csv(fn)
    else:
        print('-- reading contracts cached:', fn)
        df = pd.read_csv(fn, index_col=0)
    return df


def fetch_price_ranges(expiries, odf):
    recs = []
    for datestr in sorted(expiries):
        ddf = odf[odf['symbol'].str.contains(datestr)].sort_values('sumOpenInterestUsd', ascending=False)
        cdf = ddf[ddf['symbol'].str.contains('-C')]
        pdf = ddf[ddf['symbol'].str.contains('-P')]
        cps = [float(s.split('-')[2]) for s in cdf.head(3).symbol.values]
        pps = [float(s.split('-')[2]) for s in pdf.head(3).symbol.values]

        cps_btc = [float(s) for s in cdf.head(3).sumOpenInterest.values]
        pps_btc = [float(s) for s in pdf.head(3).sumOpenInterest.values]

        cps_dollar = [float(s) for s in cdf.head(3).sumOpenInterestUsd.values]
        pps_dollar = [float(s) for s in pdf.head(3).sumOpenInterestUsd.values]

        crange = ';'.join([f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps_btc, cps_btc))])
        drange = ';'.join([f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps_dollar, cps_dollar))])

        prange = ';'.join([f'{s[0]:,.1f} ~ {s[1]:,.1f}' for s in list(zip(pps, cps))])
        lb = np.max(pps) if len(pps) > 0 else 0
        ub = np.min(cps) if len(cps) > 0 else 0
        bd = f'{lb:,.1f} ~ {ub:,.1f}'
        print(datestr, prange, bd)

        recs += [{"expiry": datestr, "price_range": prange, "oi_qty": crange, "oi_value": drange, "bounds": bd}]
    rdf = pd.DataFrame.from_records(recs)
    return rdf


def fetch_open_interests(df, underlying, refresh_oi=False):
    expiries = list(set(df.expiry.astype(str).values))
    odf = pd.DataFrame()
    oi_fn = f"{_dir()}/_all_binance_openinterests_{underlying.lower()}.csv"

    if refresh_oi:
        oi_df = []
        for expiry in expiries:
            print('-- expiry:', expiry)
            oi = fetch_oi(expiry, underlying=underlying)
            if not oi.empty:
                oi_df += [oi]
            time.sleep(1)
        if oi_df:
            odf = oi_df = pd.concat(oi_df, axis=0)
            oi_df.to_csv(oi_fn, index=False)
    else:
        if os.path.exists(oi_fn):
            odf = pd.read_csv(oi_fn)
        else:
            print('-- use "--refresh_oi" to cache open interest data first.')
            raise Exception("Empty OI")

    odf.sumOpenInterestUsd = odf.sumOpenInterestUsd.apply(float)
    print('-- ranked all by OI:')
    print(tabulate(odf.sort_values('sumOpenInterestUsd', ascending=False).head(5), headers="keys"))

    return expiries, odf


def fetch_options_tickers(underlying="BTC"):
    endpoint = "https://eapi.binance.com/eapi/v1/ticker"
    resp = requests.get(endpoint, params={"underlyingAsset": underlying.upper()})
    if resp.status_code == 200:
        return pd.DataFrame(resp.json())
    return pd.DataFrame()


def check_put_call_parity(contracts_df, spot_bid, spot_ask, r=R_RATE, underlying="BTC"):
    now = time.time()
    tickers = fetch_options_tickers(underlying)

    if tickers.empty:
        print("Failed to fetch option tickers.")
        return pd.DataFrame()

    # Drop duplicate non-symbol columns from tickers to avoid collision during merge
    cols_to_use = [c for c in tickers.columns if c == 'symbol' or c not in contracts_df.columns]
    merged = pd.merge(contracts_df, tickers[cols_to_use], on="symbol", how="inner")

    # Explicitly derive strikePrice and expiryDate to handle cached CSV strings
    merged["strikePrice"] = merged["symbol"].apply(lambda s: float(s.split("-")[2]))
    merged["expiryDate"] = pd.to_datetime(merged["expiryDate"])
    merged["expiryTimestamp"] = merged["expiryDate"].astype('int64') // 10**9

    merged["T"] = (merged["expiryTimestamp"] - now) / (365 * 86400)
    merged = merged[merged["T"] > 0]

    calls = merged[merged["symbol"].str.endswith("-C")].copy()
    puts = merged[merged["symbol"].str.endswith("-P")].copy()

    paired = pd.merge(calls, puts, on=["expiryDate", "strikePrice", "T"], suffixes=("_call", "_put"))

    results = []
    spot_mid = (spot_bid + spot_ask) / 2.0

    for _, row in paired.iterrows():
        K = float(row["strikePrice"])
        T = float(row["T"])
        pv_K = K * np.exp(-r * T)

        c_bid = float(row.get("bidPrice_call", 0) or 0)
        c_ask = float(row.get("askPrice_call", 0) or 0)
        c_mark = float(row.get("markPrice_call", 0) or 0)

        p_bid = float(row.get("bidPrice_put", 0) or 0)
        p_ask = float(row.get("askPrice_put", 0) or 0)
        p_mark = float(row.get("markPrice_put", 0) or 0)

        # Mark Disparity: (C - P) - (S - K * e^-rT)
        mark_disparity = (c_mark - p_mark) - (spot_mid - pv_K)

        # Conversion Arbitrage: Sell Call @ Bid, Buy Put @ Ask, Buy Spot @ Ask, Borrow PV(K)
        conversion_margin = (c_bid - p_ask) - (spot_ask - pv_K)

        # Reversal Arbitrage: Buy Call @ Ask, Sell Put @ Bid, Short Spot @ Bid, Lend PV(K)
        reversal_margin = (p_bid - c_ask) - (pv_K - spot_bid)

        arb_type = "None"
        max_margin = 0.0

        if conversion_margin > 0:
            arb_type = "Conversion"
            max_margin = conversion_margin
        elif reversal_margin > 0:
            arb_type = "Reversal"
            max_margin = reversal_margin

        results.append({
            "Expiry": str(row["expiryDate"])[:10],
            "Strike": K,
            "T_Years": round(T, 4),
            "Call_Symbol": row["symbol_call"],
            "Put_Symbol": row["symbol_put"],
            "PV_K": round(pv_K, 2),
            "Mark_Disparity": round(mark_disparity, 2),
            "Conv_Margin": round(conversion_margin, 2),
            "Rev_Margin": round(reversal_margin, 2),
            "Arb_Type": arb_type,
            "Arb_Margin_USDT": round(max_margin, 2)
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        return res_df.sort_values(by="Arb_Margin_USDT", ascending=False)
    return res_df


def _wrapper_price_range(underlying, show_atm_contracts=False, update=False):
    df = refresh_contracts(underlying, update=update)
    expiries, odf = fetch_open_interests(df, underlying, refresh_oi=update)
    rdf = fetch_price_ranges(expiries, odf)

    rsp = {
        "columns": list(rdf.columns),
        "data": [list(e) for e in rdf.to_records(index=False)]
    }

    if show_atm_contracts:
        r = get_atm(underlying, df)
        rsp['atm_contracts'] = {}
        rsp['atm_contracts']['columns'] = [str(v) for v in r.keys()]
        rsp['atm_contracts']['data'] = list(r.values())

    return rsp


@click.command()
@click.option('--underlying', default="BTC")
@click.option('--update', is_flag=True, default=False, help='update contracts list')
@click.option('--refresh_oi', is_flag=True, default=False, help='update OI of contracts')
@click.option('--check_price_ranges', is_flag=True, default=False)
@click.option('--check_parity', is_flag=True, default=False, help='check Put-Call parity and arbitrage opportunities')
def main(underlying, update, refresh_oi, check_price_ranges, check_parity):
    assert underlying and len(underlying) > 0, "Must provide --underlying=<BTC|ETH|etc.>"

    df = refresh_contracts(underlying, update=update)
    expiries, odf = fetch_open_interests(df, underlying, refresh_oi=refresh_oi)

    if check_price_ranges:
        print('\n-- prices range indicated by options OI (implied by insidious market-maker, who can sell options on binance):')
        fetch_price_ranges(expiries, odf)
        sys.exit()

    if check_parity:
        print(f'\n-- Checking Put-Call Parity for {underlying.upper()} (r = {R_RATE*100:.1f}%)...')
        if os.getenv("YAHOO_LOCAL", None):
            spot_bid, spot_ask = 59000.0, 59000.0
        else:
            spot_bid, spot_ask = binance_spot(f"{underlying.upper()}/USDT")

        parity_df = check_put_call_parity(df, spot_bid, spot_ask, r=R_RATE, underlying=underlying)

        if not parity_df.empty:
            print("\n-- Ranked Put-Call Disparities & Arbitrage Opportunities:")
            print(tabulate(parity_df.head(20), headers="keys", tablefmt="grid"))

            arbs = parity_df[parity_df["Arb_Margin_USDT"] > 0]
            if not arbs.empty:
                print(f"\n-- Actionable Arbitrage Opportunities ({len(arbs)} found):")
                print(tabulate(arbs, headers="keys", tablefmt="grid"))
            else:
                print("\n-- No positive arbitrage margin opportunities found after bid/ask spread costs.")
        sys.exit()

    atm_contracts = get_atm(underlying, df)
    contracts = []
    recs = []
    for expiry, atms in atm_contracts.items():
        for atm in atms:
            contracts += [atm]
            spot_ric, T, K, ctype = extract_specs(atm)
            recs += [(spot_ric, T, K, ctype, atm,)]
    df = pd.DataFrame.from_records(recs)
    df.columns = 'spot_ric,T,K,ctype,contract'.split(',')

    _f = lambda v: f"$ {v:,.0f}" if not isinstance(v, str) else v
    df['raw_oi'] = df.contract.apply(lambda s: odf[odf.symbol == s].sumOpenInterestUsd.iloc[0])
    df['oi'] = df.raw_oi.apply(lambda s: _f(s))

    print('-- ranked ATM by OI:')
    print(tabulate(df.sort_values('raw_oi', ascending=False), headers="keys"))
    df.drop(['raw_oi'], inplace=True, axis=1)

    get_binance_spot()
    print('-- ATM by maturities:')
    print(tabulate(df, headers="keys"))
    print('  -- ATM by maturities (Puts):')
    print(tabulate(df[df.ctype == 'put'], headers="keys"))
    print('  -- ATM by maturities (Calls):')
    print(tabulate(df[df.ctype == 'call'], headers="keys"))

    fn = f"{_dir()}/_atms_{underlying.lower()}.csv"
    with open(fn, 'w') as fh:
        fh.write(','.join(contracts))
    print('-- written:', f"{_dir()}/_all_binance_contracts_{underlying.lower()}.csv")
    print('-- written:', fn)


if __name__ == '__main__':
    main()
