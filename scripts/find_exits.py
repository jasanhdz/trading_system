import re

LOG_PATH = '/home/jasan/.pm2/logs/01-Trading-Bot-out.log'

def find_exits():
    print("Scanning logs for SOL and ADA exits...")
    
    target_symbols = ['SOLUSDT', 'ADAUSDT']
    
    # Regex to capture relevant lines
    # We look for "sage_" events or "close" or "filled" or "Selling"
    # We also want to capture the timestamp
    
    relevant_lines = []
    
    try:
        with open(LOG_PATH, 'r') as f:
            # Read last 5000 lines (approx) - efficient reading
            lines = f.readlines()[-5000:]
            
            for i, line in enumerate(lines):
                # Check if line contains target symbol
                # Note: Logger might not always print symbol in the same line as the message
                # But usually "tick_start" or "Selling" includes it.
                
                # Strategy: Look for lines with target symbols AND lines with "sage_" that might be related
                # Since sage_ logs don't have symbol, we might need to infer from context (previous lines)
                # But let's first look for explicit mentions.
                
                if any(s in line for s in target_symbols):
                    relevant_lines.append(f"[{i}] {line.strip()}")
                
                # Also look for "sage_oracle_panic" or "sage_watcher_panic"
                if "sage_oracle_panic" in line or "sage_watcher_panic" in line:
                    relevant_lines.append(f"[{i}] {line.strip()}")
                    
                # Look for "Selling" or "Closing"
                if "Selling" in line or "Closing" in line or "Filled" in line:
                     if any(s in line for s in target_symbols):
                        relevant_lines.append(f"[{i}] {line.strip()}")

    except Exception as e:
        print(f"Error reading log: {e}")
        return

    print(f"Found {len(relevant_lines)} relevant lines.")
    for line in relevant_lines[-50:]: # Show last 50 matches
        print(line)

if __name__ == "__main__":
    find_exits()
