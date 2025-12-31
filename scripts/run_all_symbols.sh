#!/bin/bash
# Multi-Symbol Grid Search Runner
# Runs backtest_system_v2.py sequentially for each symbol
# Each run is independent, avoiding memory issues

VENV="/home/jasan/Develop/trading_system/.venv_rocm62/bin/python3"
SCRIPT="/home/jasan/Develop/trading_system/scripts/backtest_system_v2.py"
OUTPUT_DIR="/home/jasan/Develop/trading_system/scripts/grid_results"

# Symbols to test
SYMBOLS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "AVAXUSDT" "DOGEUSDT" "XRPUSDT" "ADAUSDT" "LINKUSDT")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "🚀 MULTI-SYMBOL GRID SEARCH RUNNER"
echo "   Started: $(date)"
echo "   Symbols: ${#SYMBOLS[@]}"
echo "========================================================================"

RESULTS_FILE="$OUTPUT_DIR/summary.txt"
echo "Grid Search Results - $(date)" > "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

for i in "${!SYMBOLS[@]}"; do
    SYMBOL="${SYMBOLS[$i]}"
    PROGRESS=$((i + 1))
    
    echo ""
    echo "📊 Progress: $PROGRESS/${#SYMBOLS[@]} - Testing $SYMBOL"
    echo "============================================================"
    
    # Run backtest for this symbol (3 days)
    $VENV $SCRIPT --symbol "$SYMBOL" --days 3 --hours 0 2>&1 | tee "$OUTPUT_DIR/${SYMBOL}_backtest.log"
    
    # Extract and save key results
    echo "" >> "$RESULTS_FILE"
    echo "=== $SYMBOL ===" >> "$RESULTS_FILE"
    tail -20 "$OUTPUT_DIR/${SYMBOL}_backtest.log" >> "$RESULTS_FILE"
    
    echo "   💾 Saved to $OUTPUT_DIR/${SYMBOL}_backtest.log"
done

echo ""
echo "========================================================================"
echo "✅ COMPLETED: $(date)"
echo "📁 Results directory: $OUTPUT_DIR"
echo "📋 Summary file: $RESULTS_FILE"
echo "========================================================================"

# Print summary
echo ""
echo "🏆 QUICK SUMMARY:"
cat "$RESULTS_FILE"
