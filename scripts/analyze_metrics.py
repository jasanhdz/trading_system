import re
import sys
import json

def parse_log(file_path):
    metrics = {}
    current_symbol = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Detect symbol start
            m_symbol = re.search(r"Training XGBoost for (\w+)", line)
            if m_symbol:
                current_symbol = m_symbol.group(1)
                continue
            
            # Detect metrics
            # [468]   train-mlogloss:0.04087  train-merror:0.00000    eval-mlogloss:0.04087   eval-merror:0.00000
            if current_symbol and "[" in line and "eval-mlogloss" in line:
                m_metrics = re.search(r"eval-mlogloss:([\d\.]+)\s+eval-merror:([\d\.]+)", line)
                if m_metrics:
                    loss = float(m_metrics.group(1))
                    error = float(m_metrics.group(2))
                    metrics[current_symbol] = {"loss": loss, "error": error}
                    
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <log_file>")
        sys.exit(1)
        
    metrics = parse_log(sys.argv[1])
    print(json.dumps(metrics, indent=2))
