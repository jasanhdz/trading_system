#!/bin/bash
# scripts/run_regime_config_grid.sh
# "Configuration Grid" para Ninja v3.0 (Camaleón Agent)
# Prueba diferentes combinaciones de Leverage/Threshold en Regimes
# modificando regime_config.json temporalmente.

set -e

# Configuración
PROJECT_DIR="/home/jasan/Develop/trading_system"
BACKTEST_SCRIPT="scripts/backtest_system_v2.py"
CONFIG_FILE="$PROJECT_DIR/binance-futures-bot-ts/regime_config.json"
BACKUP_FILE="$PROJECT_DIR/binance-futures-bot-ts/regime_config.json.bak"
REPORT_DIR="$PROJECT_DIR/reports/regime_grid"
DAYS_BACKTEST=7  # Días a probar en cada escenario

# Activar entorno
cd "$PROJECT_DIR"
source .venv_cuda/bin/activate

# Colores para consola
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "🦎 NINJA V3.0: CONFIGURATION GRID SEARCH"
echo "=================================================="
echo "   Objetivo: Encontrar la 'Personalidad' óptima del Agente."
echo "   Método: Modificar regime_config.json temporalmente."
echo "=================================================="

# 1. Verificar jq
if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ ERROR: 'jq' no está instalado.${NC}"
    echo "   Instala con: sudo apt install jq"
    exit 1
fi

# 2. Backup
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✅ Backup creado en $BACKUP_FILE${NC}"
else
    echo -e "${YELLOW}⚠️ No se encontró regime_config.json. Usando config por defecto.${NC}"
fi

# 3. Crear dir de reportes
mkdir -p "$REPORT_DIR"

# Función para ejecutar escenario
run_scenario() {
    local ID=$1
    local DESC=$2
    local SYMBOL=$3
    shift 3
    local PATCH="$@"

    echo ""
    echo "--------------------------------------------------"
    echo -e "${BLUE}SCENARIO $ID: $DESC${NC}"
    echo "--------------------------------------------------"

    # Aplicar Parche si existe
    if [ -n "$PATCH" ] && [ -f "$CONFIG_FILE" ]; then
        # Merge patch con config existente
        jq -s '.[0] * .[1]' "$BACKUP_FILE" <(echo "$PATCH") > "$CONFIG_FILE"
        echo -e "   ${GREEN}Parche aplicado${NC}"
        # Guardar config del escenario
        cp "$CONFIG_FILE" "$REPORT_DIR/scenario_${ID}_config.json"
    fi

    # Mostrar Config Activa (Resumen)
    if [ -f "$CONFIG_FILE" ]; then
        echo "   WHALE Lev: $(jq -r '.REGIMES.WHALE.leverage // "N/A"' "$CONFIG_FILE")"
        echo "   BLOODBATH Lev: $(jq -r '.REGIMES.BLOODBATH.leverage // "N/A"' "$CONFIG_FILE")"
        echo "   MONK Lev: $(jq -r '.REGIMES.MONK.leverage // "N/A"' "$CONFIG_FILE")"
    fi

    # Ejecutar Backtest
    echo -e "   ${YELLOW}Ejecutando backtest para $SYMBOL...${NC}"
    python "$BACKTEST_SCRIPT" --symbol "$SYMBOL" --days "$DAYS_BACKTEST" > "$REPORT_DIR/scenario_${ID}_output.log" 2>&1 || true
    
    # Renombrar reporte para que no se machaque
    for file in backtest_trades_${SYMBOL}_v2.csv backtest_equity_${SYMBOL}_v2.png; do
        if [ -f "$file" ]; then
            mv "$file" "$REPORT_DIR/scenario_${ID}_$(basename $file)"
            echo -e "   ${GREEN}Guardado: $REPORT_DIR/scenario_${ID}_$(basename $file)${NC}"
        fi
    done
    
    # Extraer métricas del log
    if [ -f "$REPORT_DIR/scenario_${ID}_output.log" ]; then
        WINS=$(grep -oP 'Win Rate: \K[0-9.]+' "$REPORT_DIR/scenario_${ID}_output.log" || echo "N/A")
        PF=$(grep -oP 'Profit Factor: \K[0-9.]+' "$REPORT_DIR/scenario_${ID}_output.log" || echo "N/A")
        TOTAL_PNL=$(grep -oP 'Total PnL: \K[-0-9.]+' "$REPORT_DIR/scenario_${ID}_output.log" || echo "N/A")
        echo -e "   📊 Win Rate: ${WINS}% | PF: ${PF} | PnL: ${TOTAL_PNL}"
    fi
    
    # Restaurar config original
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$CONFIG_FILE"
    fi
}

# --- DEFINICIÓN DE ESCENARIOS ---

# Símbolos a probar
SYMBOLS=("BTCUSDT" "ETHUSDT")

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "=================================================="
    echo -e "${GREEN}🎯 Testing Symbol: $SYMBOL${NC}"
    echo "=================================================="
    
    # Escenario 1: Default (Benchmark)
    run_scenario "01_DEFAULT_${SYMBOL}" "Config Actual (Benchmark)" "$SYMBOL" ""

    # Escenario 2: "Bunker Cauteloso" (Baja riesgo)
    run_scenario "02_BUNKER_SAFE_${SYMBOL}" "Modo Bunker Cauteloso (3-5x)" "$SYMBOL" '
    {
      "REGIMES": {
        "BLOODBATH": { "leverage": 5, "hard_stop_roe": -0.02 },
        "WHALE": { "leverage": 3, "hard_stop_roe": -0.25 },
        "MONK": { "leverage": 5, "hard_stop_roe": -0.05 }
      }
    }'

    # Escenario 3: "Furia Sangrienta" (High Risk Bloodbath)
    run_scenario "03_BLOOD_RAGE_${SYMBOL}" "Bloodbath 20x Agresivo" "$SYMBOL" '
    {
      "REGIMES": {
        "BLOODBATH": { "leverage": 20, "hard_stop_roe": -0.010, "entry_threshold": 0.25 },
        "WHALE": { "leverage": 5, "hard_stop_roe": -0.20 }
      }
    }'

    # Escenario 4: "Cazador de Ballenas" (Trend Following)
    run_scenario "04_WHALE_AGRO_${SYMBOL}" "Whale 10x Moonbag Mode" "$SYMBOL" '
    {
      "REGIMES": {
        "WHALE": { "leverage": 10, "hard_stop_roe": -0.20, "entry_threshold": 0.45 }
      }
    }'
done

# 4. Restore Original
if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" "$CONFIG_FILE"
    echo ""
    echo "=================================================="
    echo -e "${GREEN}✅ GRID SEARCH COMPLETADO${NC}"
    echo "=================================================="
    echo "   Reportes guardados en: $REPORT_DIR"
    echo "   Configuración original restaurada."
fi

# 5. Generar Resumen
echo ""
echo "📊 RESUMEN DE RESULTADOS:"
echo "--------------------------------------------------"
for log in "$REPORT_DIR"/scenario_*_output.log; do
    if [ -f "$log" ]; then
        SCENARIO=$(basename "$log" | sed 's/_output.log//')
        WINS=$(grep -oP 'Win Rate: \K[0-9.]+' "$log" 2>/dev/null || echo "N/A")
        PF=$(grep -oP 'Profit Factor: \K[0-9.]+' "$log" 2>/dev/null || echo "N/A")
        TOTAL=$(grep -oP 'Total PnL: \K[-0-9.]+' "$log" 2>/dev/null || echo "N/A")
        printf "   %-35s | WR: %5s%% | PF: %5s | PnL: %10s\n" "$SCENARIO" "$WINS" "$PF" "$TOTAL"
    fi
done

echo ""
echo "🦎 ¡Que gane el mejor Camaleón!"
