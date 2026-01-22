import pandas as pd

def check_data():
    try:
        df = pd.read_csv('data/phantom_features.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("Row 220-225:")
        subset = df.iloc[220:226]
        for idx, row in subset.iterrows():
            print(f"Index: {idx} | Time: {row['timestamp']} | High: {row['high_eth']} | Low: {row['low_eth']} | Close: {row['close_eth']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()
