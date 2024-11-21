import pandas as pd,os

from crossing_check import calculate_price_cross_counts

df = pd.read_csv(os.getenv("USER_HOME",'') + '/tmp/perp_solusdt_15m.csv')
df['timestamp'] = df['starttime'].apply(pd.Timestamp)

result = calculate_price_cross_counts(df, w=3, use_weight=False)
print(result)

