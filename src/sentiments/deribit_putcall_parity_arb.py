import datetime
import os
import sys
import time
import click
import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

R_RATE = 0.02  # Risk-free rate (2.0%)
DERIBIT_OPTION_TAKER_FEE_RATE = 0.0003  # 0.03% of underlying asset
OPTION_FEE_CAP = 0.125                 # Fee capped at 12.5% of option price
DERIBIT_FUTURES_TAKER_FEE_RATE = 0.0005 # Futures taker fee (0.05%)

BASE_URL = "https://www.deribit.com/api/v2/public"


def _dir():
    """Deribit-specific cache directory."""
    fdir = os.getenv("USER_HOME", "") + '/tmp/deribit_options/'
    fdir += datetime.datetime.strftime(datetime.datetime.today(), '%Y_%m_%d')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    return fdir


def fetch_deribit_index(underlying="BTC"):
    """Fetch current underlying index price from Deribit."""
    index_name = f"{underlying.lower()}_usd"
    try:
        resp = requests.get(f"{BASE_URL}/get_index_price", params={"index_name": index_name}).json()
        price = float(resp['result']['index_price'])
        return price, price
    except Exception as e:
        print(f"*** Failed to fetch Deribit index price for {underlying}: {e}")
        return 0.0, 0.0


def fetch_futures_tickers(underlying="BTC"):
    """Fetch market summary tickers for all active futures/forwards."""
    try:
        resp = requests.get(
            f"{BASE_URL}/get_book_summary_by_currency",
            params={"currency": underlying.upper(), "kind": "future"}
        ).json()

        summaries = resp.get("result", [])
        if not summaries:
            return {}

        df = pd.DataFrame(summaries)
        bid_col = "bid_price" if "bid_price" in df.columns else "best_bid_price"
        ask_col = "ask_price" if "ask_price" in df.columns else "best_ask_price"
        mark_col = "mark_price" if "mark_price" in df.columns else "mark_price"

        futures_map = {}
        for _, row in df.iterrows():
            sym = row.get("instrument_name")
            if not sym:
                continue
            futures_map[sym] = {
                "bid": float(row.get(bid_col, 0.0) or 0.0),
                "ask": float(row.get(ask_col, 0.0) or 0.0),
                "mark": float(row.get(mark_col, 0.0) or 0.0),
            }
        return futures_map
    except Exception as e:
        print(f"*** Failed to fetch Deribit futures tickers: {e}")
        return {}


def fetch_contracts(underlying="BTC"):
    """Fetch active option instruments for currency from Deribit."""
    try:
        resp = requests.get(
            f"{BASE_URL}/get_instruments",
            params={"currency": underlying.upper(), "kind": "option", "expired": "false"}
        ).json()
        
        instruments = resp.get("result", [])
        if not instruments:
            return pd.DataFrame()

        df = pd.DataFrame(instruments)
        df.rename(columns={"instrument_name": "symbol"}, inplace=True)
        
        df["expiryDate"] = df["expiration_timestamp"].apply(
            lambda ts: pd.Timestamp(datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc))
        )
        df["strikePrice"] = df["strike"].astype(float)
        df["tickSize"] = df["tick_size"].astype(float)
        df["expiry"] = df["symbol"].apply(lambda s: s.split("-")[1])
        
        df = df.sort_values(["expiryDate", "symbol", "strikePrice"], ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"*** Failed to fetch Deribit contracts: {e}")
        return pd.DataFrame()


def refresh_contracts(underlying, update=False):
    fn = f"{_dir()}/_all_deribit_contracts_{underlying.lower()}.csv"
    if update or not os.path.exists(fn):
        df = fetch_contracts(underlying)
        if not df.empty:
            df.to_csv(fn, index=False)
    else:
        print('-- reading cached contracts:', fn)
        df = pd.read_csv(fn)
    return df


def fetch_options_tickers(underlying="BTC"):
    """Fetch market summary tickers for all options in currency."""
    try:
        resp = requests.get(
            f"{BASE_URL}/get_book_summary_by_currency",
            params={"currency": underlying.upper(), "kind": "option"}
        ).json()

        summaries = resp.get("result", [])
        if not summaries:
            return pd.DataFrame()

        df = pd.DataFrame(summaries)
        df.rename(columns={"instrument_name": "symbol"}, inplace=True)

        bid_col = "bid_price" if "bid_price" in df.columns else "best_bid_price"
        ask_col = "ask_price" if "ask_price" in df.columns else "best_ask_price"
        mark_col = "mark_price" if "mark_price" in df.columns else "mark_price"

        u_price = df["underlying_price"].fillna(0.0).astype(float) if "underlying_price" in df.columns else 0.0

        df["bidPrice"] = df[bid_col].fillna(0.0).astype(float) * u_price if bid_col in df.columns else 0.0
        df["askPrice"] = df[ask_col].fillna(0.0).astype(float) * u_price if ask_col in df.columns else 0.0
        df["markPrice"] = df[mark_col].fillna(0.0).astype(float) * u_price if mark_col in df.columns else 0.0

        if "open_interest" in df.columns:
            df["sumOpenInterestUsd"] = df["open_interest"].fillna(0.0).astype(float) * u_price
        else:
            df["sumOpenInterestUsd"] = 0.0

        return df
    except Exception as e:
        print(f"*** Failed to fetch Deribit tickers: {e}")
        return pd.DataFrame()


def fetch_open_interests(df, underlying, refresh_oi=False):
    expiries = list(set(df.expiry.astype(str).values))
    oi_fn = f"{_dir()}/_all_deribit_openinterests_{underlying.lower()}.csv"

    if refresh_oi or not os.path.exists(oi_fn):
        tickers = fetch_options_tickers(underlying)
        if not tickers.empty:
            tickers.to_csv(oi_fn, index=False)
            odf = tickers
        else:
            odf = pd.DataFrame()
    else:
        odf = pd.read_csv(oi_fn)

    if not odf.empty and "sumOpenInterestUsd" in odf.columns:
        print('-- Top 5 Ranked by OI (USD):')
        print(tabulate(odf.sort_values('sumOpenInterestUsd', ascending=False).head(5), headers="keys"))

    return expiries, odf


def check_put_call_parity(contracts_df, spot_bid, spot_ask, r=R_RATE, underlying="BTC",
                           min_t_days=2.0 / 365.0, max_relative_spread=0.50):
    now = time.time()
    tickers = fetch_options_tickers(underlying)
    futures_map = fetch_futures_tickers(underlying)

    if tickers.empty:
        print("Failed to fetch Deribit option tickers.")
        return pd.DataFrame()

    cols_to_use = [c for c in tickers.columns if c == 'symbol' or c not in contracts_df.columns]
    merged = pd.merge(contracts_df, tickers[cols_to_use], on="symbol", how="inner")

    merged["strikePrice"] = merged["symbol"].apply(lambda s: float(s.split("-")[2]))
    merged["expiryDate"] = pd.to_datetime(merged["expiryDate"])
    merged["expiryTimestamp"] = merged["expiryDate"].astype('int64') // 10**9

    merged["T"] = (merged["expiryTimestamp"] - now) / (365 * 86400)
    merged = merged[merged["T"] >= min_t_days]

    calls = merged[merged["symbol"].str.endswith("-C")].copy()
    puts = merged[merged["symbol"].str.endswith("-P")].copy()

    paired = pd.merge(calls, puts, on=["expiryDate", "strikePrice", "T"], suffixes=("_call", "_put"))

    results = []
    spot_mid = (spot_bid + spot_ask) / 2.0 if (spot_bid and spot_ask) else 0.0

    for _, row in paired.iterrows():
        K = float(row["strikePrice"])
        T = float(row["T"])

        # Match maturity delivery future (e.g. BTC-25JUN27)
        u_index = str(row.get("underlying_index_call", "") or row.get("underlying_index_put", ""))
        
        if u_index in futures_map and futures_map[u_index]["mark"] > 0:
            f_bid = futures_map[u_index]["bid"]
            f_ask = futures_map[u_index]["ask"]
            f_mark = futures_map[u_index]["mark"]
        else:
            # Fallback to option row's underlying price or spot
            f_mark = float(row.get("underlying_price_call", spot_mid) or spot_mid)
            f_bid = spot_bid if spot_bid > 0 else f_mark
            f_ask = spot_ask if spot_ask > 0 else f_mark

        f_mid = (f_bid + f_ask) / 2.0 if (f_bid > 0 and f_ask > 0) else f_mark

        if not (0.70 <= (K / f_mid) <= 1.30):
            continue

        pv_K = K * np.exp(-r * T)

        c_bid = float(row.get("bidPrice_call", 0) or 0)
        c_ask = float(row.get("askPrice_call", 0) or 0)
        c_mark = float(row.get("markPrice_call", 0) or 0)

        p_bid = float(row.get("bidPrice_put", 0) or 0)
        p_ask = float(row.get("askPrice_put", 0) or 0)
        p_mark = float(row.get("markPrice_put", 0) or 0)

        # Forward-adjusted Parity Calculations
        # Conversion: Long Future (F_ask), Long Put (P_ask), Short Call (C_bid)
        raw_conv = (c_bid - p_ask) - (f_ask - pv_K) if (c_bid > 0 and p_ask > 0 and f_ask > 0) else -9999.0
        # Reversal: Short Future (F_bid), Short Put (P_bid), Long Call (C_ask)
        raw_rev = (p_bid - c_ask) - (pv_K - f_bid) if (p_bid > 0 and c_ask > 0 and f_bid > 0) else -9999.0

        has_surface_signal = (raw_conv > 0) or (raw_rev > 0) or (c_ask <= 0 and p_bid > 0) or (p_ask <= 0 and c_bid > 0)
        invalid_quotes = (c_bid <= 0 or c_ask <= 0 or p_bid <= 0 or p_ask <= 0 or c_ask <= c_bid or p_ask <= p_bid)

        c_spread_ratio = (c_ask - c_bid) / max(c_mark, 1e-4) if not invalid_quotes else 1.0
        p_spread_ratio = (p_ask - p_bid) / max(p_mark, 1e-4) if not invalid_quotes else 1.0
        excessive_spread = (c_spread_ratio > max_relative_spread or p_spread_ratio > max_relative_spread)

        c_ask_fee = min(f_mid * DERIBIT_OPTION_TAKER_FEE_RATE, OPTION_FEE_CAP * c_ask) if c_ask > 0 else 0
        c_bid_fee = min(f_mid * DERIBIT_OPTION_TAKER_FEE_RATE, OPTION_FEE_CAP * c_bid) if c_bid > 0 else 0
        p_ask_fee = min(f_mid * DERIBIT_OPTION_TAKER_FEE_RATE, OPTION_FEE_CAP * p_ask) if p_ask > 0 else 0
        p_bid_fee = min(f_mid * DERIBIT_OPTION_TAKER_FEE_RATE, OPTION_FEE_CAP * p_bid) if p_bid > 0 else 0

        future_ask_fee = f_ask * DERIBIT_FUTURES_TAKER_FEE_RATE
        future_bid_fee = f_bid * DERIBIT_FUTURES_TAKER_FEE_RATE

        conv_fees = c_bid_fee + p_ask_fee + future_ask_fee
        rev_fees = p_bid_fee + c_ask_fee + future_bid_fee

        net_conv = raw_conv - conv_fees if not invalid_quotes else -9999.0
        net_rev = raw_rev - rev_fees if not invalid_quotes else -9999.0

        mark_disparity = (c_mark - p_mark) - (f_mid - pv_K)

        if not invalid_quotes and not excessive_spread and net_conv > 0:
            results.append({
                "Expiry": str(row["expiryDate"])[:10],
                "Strike": K,
                "T_Years": round(T, 4),
                "F_T_Mark": round(f_mark, 2),
                "Call_Symbol": row["symbol_call"],
                "Put_Symbol": row["symbol_put"],
                "PV_K": round(pv_K, 2),
                "Mark_Disparity": round(mark_disparity, 2),
                "Raw_Margin": round(raw_conv, 2),
                "Net_Arb_Margin": round(net_conv, 2),
                "Arb_Type": "Conversion",
                "Status": "Actionable",
                "Reject_Reason": "None (Executable)"
            })
        elif not invalid_quotes and not excessive_spread and net_rev > 0:
            results.append({
                "Expiry": str(row["expiryDate"])[:10],
                "Strike": K,
                "T_Years": round(T, 4),
                "F_T_Mark": round(f_mark, 2),
                "Call_Symbol": row["symbol_call"],
                "Put_Symbol": row["symbol_put"],
                "PV_K": round(pv_K, 2),
                "Mark_Disparity": round(mark_disparity, 2),
                "Raw_Margin": round(raw_rev, 2),
                "Net_Arb_Margin": round(net_rev, 2),
                "Arb_Type": "Reversal",
                "Status": "Actionable",
                "Reject_Reason": "None (Executable)"
            })
        elif has_surface_signal:
            reasons = []
            if invalid_quotes:
                if c_ask <= 0 or p_ask <= 0 or c_bid <= 0 or p_bid <= 0:
                    reasons.append("Zero/Empty Quote")
                elif c_ask <= c_bid or p_ask <= p_bid:
                    reasons.append("Crossed Order Book")
            if excessive_spread:
                reasons.append("Spread > 50%")
            if not invalid_quotes and not excessive_spread and (net_conv <= 0 and net_rev <= 0):
                reasons.append("Fee/Spread Drag")

            raw_disp = max(raw_conv, raw_rev) if max(raw_conv, raw_rev) > -9000 else abs(mark_disparity)

            results.append({
                "Expiry": str(row["expiryDate"])[:10],
                "Strike": K,
                "T_Years": round(T, 4),
                "F_T_Mark": round(f_mark, 2),
                "Call_Symbol": row["symbol_call"],
                "Put_Symbol": row["symbol_put"],
                "PV_K": round(pv_K, 2),
                "Mark_Disparity": round(mark_disparity, 2),
                "Raw_Margin": round(raw_disp, 2),
                "Net_Arb_Margin": 0.0,
                "Arb_Type": "Conversion" if raw_conv > raw_rev else "Reversal",
                "Status": "Phantom",
                "Reject_Reason": ", ".join(reasons) if reasons else "Unactionable"
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        return res_df.sort_values(by=["Status", "Net_Arb_Margin", "Raw_Margin"], ascending=[True, False, False])
    return res_df


@click.command()
@click.option('--underlying', default="BTC")
@click.option('--update', is_flag=True, default=False, help='Update cached contract list')
@click.option('--refresh_oi', is_flag=True, default=False, help='Update open interest data')
@click.option('--check_parity', is_flag=True, default=False, help='Check Deribit Put-Call parity opportunities')
def main(underlying, update, refresh_oi, check_parity):
    df = refresh_contracts(underlying, update=update)
    if df.empty:
        print("No contracts loaded.")
        sys.exit(1)

    expiries, odf = fetch_open_interests(df, underlying, refresh_oi=refresh_oi)

    if check_parity:
        print(f'\n-- Checking Forward-Adjusted Deribit Put-Call Parity for {underlying.upper()} (r = {R_RATE*100:.1f}%)...')
        spot_bid, spot_ask = fetch_deribit_index(underlying)
        
        parity_df = check_put_call_parity(df, spot_bid, spot_ask, r=R_RATE, underlying=underlying)

        if not parity_df.empty:
            actionable_df = parity_df[parity_df["Status"] == "Actionable"]
            phantom_df = parity_df[parity_df["Status"] == "Phantom"]

            if not actionable_df.empty:
                print(f"\n-- Actionable Arbitrage Opportunities ({len(actionable_df)} found):")
                print(tabulate(actionable_df, headers="keys", tablefmt="grid"))
            else:
                print("\n-- No actionable arbitrage margin opportunities found after forward adjustment, spreads, and fees.")

            if not phantom_df.empty:
                print(f"\n-- Detected Phantom / Unactionable Arbitrage Signals ({len(phantom_df)} sampled):")
                print(tabulate(phantom_df.head(15), headers="keys", tablefmt="grid"))
        else:
            print("\n-- No valid option pairs found.")


if __name__ == '__main__':
    main()
