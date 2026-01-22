import pandas as pd

df = pd.read_csv('data/phantom_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

start_time = pd.Timestamp('2025-01-16 20:35:00')
end_time = pd.Timestamp('2025-01-17 20:00:00')

mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
subset = df[mask]

target_price = 3263.7
drops = subset[subset['low_eth'] <= target_price]

print(f"Candles with Low <= {target_price}:")
print(drops[['timestamp', 'open_eth', 'high_eth', 'low_eth', 'close_eth']])
