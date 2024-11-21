import pandas as pd
import numpy as np

def calculate_price_cross_counts(df, w=3, price_step=10, num_levels=10, use_weight=False):
    lowest_price = df["low"].min()
    highest_price = df["high"].max()

    if num_levels>0: # Overwrite
        price_levels = np.linspace(lowest_price, highest_price, num_levels)
    else:
        price_levels = np.arange( lowest_price, highest_price, price_step)  

    # Add a column for the window end time
    df["window_end"] = df["timestamp"] + pd.to_timedelta(w, unit="h")

    # Initialize a DataFrame to hold crossing counts for all price levels
    crossing_matrix = pd.DataFrame(0, index=df.index, columns=price_levels)

    # Check for price crosses for each level
    for price in price_levels:
        is_crossed = (df["low"] <= price) & (price <= df["high"])
        crossing_matrix[price] = is_crossed.astype(int)

    if use_weight:
        # Weight crossing counts by volume
        weighted_crossing_matrix = crossing_matrix.mul(df["volume"], axis=0)

        # Normalize by dividing by total volume
        total_volume = df["volume"].sum()
        if total_volume > 0:
            weighted_crossing_matrix = weighted_crossing_matrix / total_volume
    else:
        # If not weighted, just use the crossing matrix
        weighted_crossing_matrix = crossing_matrix

    # Expand weighted_crossing_matrix to consider sliding windows
    weighted_crossing_counts = (
        weighted_crossing_matrix.T.dot(
            pd.get_dummies(df.index).rolling(window=w, min_periods=1).sum()
        )
    )

    # Sum weighted counts for each price level across all time windows
    final_counts = weighted_crossing_counts.sum(axis=1)

    if use_weight:
        final_counts *= 100
        result = pd.DataFrame({"Price": final_counts.index, "Weighted Cross Count": final_counts.values})
    else:
        result = pd.DataFrame({"Price": final_counts.index, "Cross Count": final_counts.values})

    return result

