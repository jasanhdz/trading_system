#!/bin/bash
# Sequential Grid Search for All Priority Symbols
# Runs ONE symbol at a time to avoid memory issues
# Each symbol tests 15 configurations (5 thresholds × 3 stops)

VENV="/home/jasan/Develop/trading_system/.venv_rocm62/bin/python3"
SCRIPT="/home/jasan/Develop/trading_system/scripts/grid_search_single_symbol.py"
OUTPUT_DIR="/home/jasan/Develop/trading_system/scripts/grid_search_results"

# Priority symbols from .env backup (excluding BTC, DOGE, XRP which are already deployed)
SYMBOLS=(
    "ADAUSDT"
    "BNBUSDT"
    "ATOMUSDT"
    "POLUSDT"
    "BTCUSDT"
)

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/grid_search_summary.txt"

echo "========================================================================"
echo "🚀 SEQUENTIAL GRID SEARCH - ALL PRIORITY SYMBOLS"
echo "   Started: $(date)"
echo "   Symbols: ${#SYMBOLS[@]}"
echo "   Configs per symbol: 15 (5 thresholds × 3 stops)"
echo "========================================================================"

echo "Grid Search Summary - $(date)" > "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

TOTAL=${#SYMBOLS[@]}
for i in "${!SYMBOLS[@]}"; do
    SYMBOL="${SYMBOLS[$i]}"
    PROGRESS=$((i + 1))
    
    echo ""
    echo "========================================================================"
    echo "📊 PROGRESS: $PROGRESS/$TOTAL"
    echo "🎯 SYMBOL: $SYMBOL"
    echo "   Started: $(date)"
    echo "========================================================================"
    
    # Run grid search for this symbol
    $VENV $SCRIPT --symbol "$SYMBOL" --days 3
    
    # Extract best result for summary
    if [ -f "$OUTPUT_DIR/${SYMBOL}_grid_results.csv" ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "=== $SYMBOL ===" >> "$SUMMARY_FILE"
        # Get best result (highest return from CSV)
        tail -1 "$OUTPUT_DIR/${SYMBOL}_grid_results.csv" >> "$SUMMARY_FILE"
        echo "   ✅ Completed: $(date)"
    else
        echo "   ❌ No results file generated"
        echo "=== $SYMBOL === ERROR: No results" >> "$SUMMARY_FILE"
    fi
    
    # Brief pause between symbols to free memory
    sleep 5
done

echo ""
echo "========================================================================"
echo "🏆 GRID SEARCH COMPLETE"
echo "   Finished: $(date)"
echo "========================================================================"
echo ""
echo "📊 QUICK SUMMARY:"
echo "========================================================================"

# Parse and display best config for each symbol
for SYMBOL in "${SYMBOLS[@]}"; do
    CSV="$OUTPUT_DIR/${SYMBOL}_grid_results.csv"
    if [ -f "$CSV" ]; then
        # Get best result (sort by return_pct descending)
        BEST=$(tail -n +2 "$CSV" | sort -t',' -k4 -rn | head -1)
        if [ -n "$BEST" ]; then
            THR=$(echo "$BEST" | cut -d',' -f2)
            STOP=$(echo "$BEST" | cut -d',' -f3)
            RET=$(echo "$BEST" | cut -d',' -f4 | cut -c1-7)
            WR=$(echo "$BEST" | cut -d',' -f5)
            TRADES=$(echo "$BEST" | cut -d',' -f7)
            echo "$SYMBOL: Thr=$THR Stop=$STOP Return=${RET}% WR=$(echo "$WR * 100" | bc)% Trades=$TRADES"
        fi
    fi
done

echo ""
echo "📁 Detailed results: $OUTPUT_DIR/"
echo "📋 Summary: $SUMMARY_FILE"
