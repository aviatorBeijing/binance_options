import pandas as pd,os
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# RSI calculation function
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

sym = 'flm'
sym = 'deep'
sym = 'rez'
sym = 'bsw'
sym = 'alpaca'

df = pd.read_csv(os.getenv("USER_HOME","")+f'/tmp/perp_{sym.lower()}usdt_5m.csv') 

def rolling_percentile(series, window, quantile_val):
    return series.shift(1).rolling(window=window, min_periods=50).quantile(quantile_val)

def find_spikes(df, p=0.95, window=150, cooldown=15):
    df['volume_95'] = rolling_percentile(df['volume'], window, p)
    df['taker_95'] = rolling_percentile(df['taker_quote_volume'], window, p)

    bullish_indices = []
    bearish_indices = []

    i = 0
    while i < len(df):
        is_spike = (df.loc[i, 'volume'] > df.loc[i, 'volume_95']) or \
                   (df.loc[i, 'taker_quote_volume'] > df.loc[i, 'taker_95'])

        if is_spike:
            if df.loc[i, 'close'] > df.loc[i, 'open']:
                bullish_indices.append(i)
                i += cooldown
            elif df.loc[i, 'open'] > df.loc[i, 'close']:
                bearish_indices.append(i)
                i += cooldown
            else:
                i += 1
        else:
            i += 1

    return bullish_indices, bearish_indices

def simulate_trades(df, bullish_indices, bearish_indices, hold_period=10):
    trades = []

    # Long Trades
    for idx in bullish_indices:
        if idx + hold_period < len(df):
            entry_time = df.loc[idx, 'timestamp']
            exit_time = df.loc[idx + hold_period, 'timestamp']
            buy_price = df.loc[idx, 'close']
            sell_price = df.loc[idx + hold_period, 'close']
            returns = (sell_price - buy_price) / buy_price

            trades.append({
                'type': 'LONG',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': buy_price,
                'exit_price': sell_price,
                'return': returns
            })

    # Short Trades
    for idx in bearish_indices:
        if idx + hold_period < len(df):
            entry_time = df.loc[idx, 'timestamp']
            exit_time = df.loc[idx + hold_period, 'timestamp']
            sell_price = df.loc[idx, 'close']
            buy_price = df.loc[idx + hold_period, 'close']
            returns = (sell_price - buy_price) / sell_price

            trades.append({
                'type': 'SHORT',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': sell_price,
                'exit_price': buy_price,
                'return': returns
            })

    # Print trade summary
    print("\nTrade Summary:")
    for trade in trades:
        action = 'BUY' if trade['type'] == 'LONG' else 'SELL (SHORT)'
        close_action = 'SELL' if trade['type'] == 'LONG' else 'BUY TO CLOSE'
        print(f"[{trade['entry_time']}] {action} at {trade['entry_price']:.4f}")
        print(f"[{trade['exit_time']}] {close_action} at {trade['exit_price']:.4f} → Return: {trade['return']:.1%}\n")

    return trades

def plot_data(df, bullish_indices, bearish_indices, trades):
    df['rsi'] = compute_rsi(df['close'])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Price (Open & Close)
    ax1.plot(df['timestamp'], df['open'], color='blue', label='Open', alpha=0.7)
    ax1.plot(df['timestamp'], df['close'], color='orange', label='Close')
    ax1.set_title('Open & Close Prices')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: RSI
    ax2.plot(df['timestamp'], df['rsi'], label='RSI (14)', color='purple')
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.set_title('RSI')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Volume
    ax3.bar(df['timestamp'], df['volume'], width=0.01, color='gray', label='Volume')
    ax3.set_title('Volume')
    ax3.set_ylabel('Volume')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Aggregated Returns (Bullish & Bearish)
    bullish_returns = [trade['return'] for trade in trades if trade['type'] == 'LONG']
    bearish_returns = [trade['return'] for trade in trades if trade['type'] == 'SHORT']

    # Create timestamps corresponding to the trades
    bullish_timestamps = [df.loc[df['timestamp'] == trade['entry_time'], 'timestamp'].iloc[0] for trade in trades if trade['type'] == 'LONG']
    bearish_timestamps = [df.loc[df['timestamp'] == trade['entry_time'], 'timestamp'].iloc[0] for trade in trades if trade['type'] == 'SHORT']

    # Calculate cumulative returns
    cumulative_bullish = pd.Series(bullish_returns).cumsum()
    cumulative_bearish = pd.Series(bearish_returns).cumsum()

    # Plot cumulative returns for bullish and bearish
    ax4.plot(bullish_timestamps, cumulative_bullish, color='green', label='Bullish Returns')
    ax4.plot(bearish_timestamps, cumulative_bearish, color='red', label='Bearish Returns')

    ax4.set_title('Cumulative Returns (Bullish & Bearish)')
    ax4.set_ylabel('Cumulative Return')
    ax4.legend()
    ax4.grid(True)

    # Draw vertical lines for spikes in price (Bullish = green, Bearish = red)
    for idx in bullish_indices:
        t = df.loc[idx, 'timestamp']
        ax1.axvline(x=t, color='green', linestyle='--', alpha=0.7)
        ax2.axvline(x=t, color='green', linestyle='--', alpha=0.7)
        ax3.axvline(x=t, color='green', linestyle='--', alpha=0.7)

    for idx in bearish_indices:
        t = df.loc[idx, 'timestamp']
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=t, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fn = os.getenv("USER_HOME", "")+f"/tmp/meme_jumps_detection_{sym.upper()}.pdf"
    plt.savefig( fn )
    print('-- Saved:', fn )

# --- MAIN EXECUTION ---
# Assumes df is already loaded and contains OHLCV + taker_quote_volume + starttime
df['timestamp'] = pd.to_datetime(df['starttime'], unit='ms')
bullish_indices, bearish_indices = find_spikes(df, p=0.68, window=150, cooldown=10)
trades = simulate_trades(df, bullish_indices, bearish_indices, hold_period=30)
plot_data(df, bullish_indices, bearish_indices, trades)

