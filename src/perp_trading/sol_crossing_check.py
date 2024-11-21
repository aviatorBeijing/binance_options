import pandas as pd

from crossing_check import calculate_price_cross_counts

df = pd.read_csv('/Users/junma/tmp/perp_solusdt_15m.csv')
df['timestamp'] = df['starttime'].apply(pd.Timestamp)

result = calculate_price_cross_counts(df, use_weight=True)
print(result)

