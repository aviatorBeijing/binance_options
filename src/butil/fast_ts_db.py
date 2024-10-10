import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Define the database and table name
DB_NAME = "timeseries_data.db"
TABLE_NAME = "timeseries"

# Define the schema for the timeseries table
CREATE_TABLE_QUERY = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    timestamp TEXT PRIMARY KEY,
    symbol TEXT,
    bid REAL,
    ask REAL,
    spread REAL,
    percentile_68_spread REAL
);
"""

# Initialize the database and create table if not exists
def initialize_db(db_name=DB_NAME, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_name)
    conn.execute(CREATE_TABLE_QUERY)
    conn.commit()
    conn.close()

# Write a new set of data points into the database
def write_data_to_db(df, db_name=DB_NAME, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_name)
    
    # Remove duplicate entries based on timestamp and symbol
    df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
    
    # Write data to the database
    df.to_sql(table_name, conn, if_exists='append', index=False)
    
    # Keep only 4 hours of data: delete data older than the last 4 hours
    cursor = conn.cursor()
    
    # Calculate the cutoff time for the 4-hour window
    cursor.execute(f"SELECT MAX(timestamp) FROM {table_name}")
    latest_timestamp = cursor.fetchone()[0]
    
    if latest_timestamp:
        cutoff_time = datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S") - timedelta(hours=4)
        cursor.execute(f"DELETE FROM {table_name} WHERE timestamp < ?", (cutoff_time.strftime("%Y-%m-%d %H:%M:%S"),))
    
    conn.commit()
    conn.close()

# Read and return data from the database, ordered by timestamp
def read_data_from_db(db_name=DB_NAME, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_name)
    
    query = f"SELECT * FROM {table_name} ORDER BY timestamp"
    df = pd.read_sql(query, conn)
    
    conn.close()
    return df

# Generate a sample snippet of data for testing
def generate_sample_data(start_time, symbol="AAPL", num_points=300):
    time_intervals = pd.date_range(start=start_time, periods=num_points, freq='7S')
    bid_prices = pd.Series(range(num_points)) + 100
    ask_prices = bid_prices + 0.5
    spread = ask_prices - bid_prices
    percentile_68_spread = spread * 1.02  # Just a placeholder calculation

    return pd.DataFrame({
        'timestamp': time_intervals.strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': [symbol] * num_points,
        'bid': bid_prices,
        'ask': ask_prices,
        'spread': spread,
        'percentile_68_spread': percentile_68_spread
    })

# Main script to test the API
if __name__ == "__main__":
    # Initialize the database and table
    initialize_db()

    # Generate a sample snippet of data (4 hours window, data points every ~7 seconds)
    sample_df = generate_sample_data(start_time=datetime.now() - timedelta(hours=4))

    # Write the sample data to the database
    write_data_to_db(sample_df)

    # Read the data back and display it
    retrieved_df = read_data_from_db()
    print(retrieved_df)

