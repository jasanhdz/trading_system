#!/usr/bin/env python3
"""
NINJA SYSTEM v4.0: Symbol-Specific Regime Grid Search
======================================================
Objetivo: Encontrar la mejor personalidad (Escenario) para cada símbolo.

Ejecutar: python scripts/symbol_grid_search.py [--symbols BTCUSDT,ETHUSDT] [--days 7]
"""

import sys
import yaml
import subprocess
import json
import os
import copy
import argparse
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
# 1. PATHS CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
BOT_DIR = PROJECT_ROOT / "binance-futures-bot-ts"

BASE_CONFIG_FILE = BOT_DIR / "regime_config.yaml"
TEMP_CONFIG_FILE = BOT_DIR / "regime_config.grid_temp.yaml"
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_system_v2.py"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ═════════════════════════════════════════════════════════════════════════════
# 2. SYMBOLS FROM .ENV
# ═════════════════════════════════════════════════════════════════════════════

def get_symbols_from_env() -> list:
    """Read symbols from bot's .env file"""
    env_file = BOT_DIR / ".env"
    if not env_file.exists():
        print(f"[WARN] .env not found, using fallback symbols")
        return ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
    
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith("SYMBOLS="):
                # Parse: DOGEUSDT:10:0.75:1h,LINKUSDT:10:0.65:1h,...
                raw = line.split("=", 1)[1].strip()
                symbols = [s.split(":")[0] for s in raw.split(",")]
                return symbols
    
    return ["BTCUSDT", "ETHUSDT"]

DEFAULT_SYMBOLS = get_symbols_from_env()

# ═════════════════════════════════════════════════════════════════════════════
# 3. ESCENARIOS (MATRIZ DE PARÁMETROS)
# ═════════════════════════════════════════════════════════════════════════════
# Cada escenario define cómo sobrescribir la configuración base para un símbolo.
# El escenario ganador será aplicado como SYMBOL_OVERRIDE en regime_config.live.yaml

SCENARIOS = {
    "DEFAULT": {
        # Sin cambios - usa la config base
    },
    
    "TORTUGA": {
        # Más conservador: menos leverage, stops más anchos
        "BLOODBATH": { "leverage": 10, "hard_stop_roe": -0.020 },
        "WHALE":     { "leverage": 3,  "hard_stop_roe": -0.25 },
        "MONK":      { "leverage": 5,  "hard_stop_roe": -0.06 }
    },
    
    "SANGRIENTO": {
        # Agresivo: más leverage, stops muy ajustados
        "BLOODBATH": { "leverage": 20, "hard_stop_roe": -0.010, "entry_threshold": 0.25 },
        "WHALE":     { "leverage": 8,  "hard_stop_roe": -0.15,  "entry_threshold": 0.40 }
    },
    
    "CAZADOR": {
        # Entrada agresiva: umbral más bajo para entrar más seguido
        "WHALE": { "entry_threshold": 0.45 },
        "MONK":  { "entry_threshold": 0.30 }
    }
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_yaml(filepath: Path) -> dict:
    """Load YAML configuration file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(filepath: Path, data: dict):
    """Save YAML configuration file with flow style for regimes"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def apply_scenario_to_config(base_config: dict, symbol: str, scenario: dict) -> dict:
    """Apply a scenario's overrides to the config for a specific symbol"""
    config = copy.deepcopy(base_config)
    
    if not scenario:  # DEFAULT scenario
        # Remove any existing overrides for this symbol
        if "SYMBOL_OVERRIDES" in config and symbol in config.get("SYMBOL_OVERRIDES", {}):
            del config["SYMBOL_OVERRIDES"][symbol]
        return config
    
    # Initialize SYMBOL_OVERRIDES if not exists
    if "SYMBOL_OVERRIDES" not in config:
        config["SYMBOL_OVERRIDES"] = {}
    
    # Apply scenario overrides for this symbol
    config["SYMBOL_OVERRIDES"][symbol] = copy.deepcopy(scenario)
    
    return config

def parse_backtest_results(output: str) -> dict:
    """Parse backtest output and extract key metrics"""
    results = {
        "final_capital": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "pnl": 0.0,
        "raw_output": output[-500:] if len(output) > 500 else output  # Keep last 500 chars
    }
    
    for line in output.split('\n'):
        line_lower = line.lower()
        
        # Capital Final: $10234.56
        if "capital final" in line_lower or "final capital" in line_lower:
            try:
                parts = line.split('$')
                if len(parts) > 1:
                    results["final_capital"] = float(parts[1].strip().replace(',', ''))
            except (ValueError, IndexError):
                pass
        
        # Total Trades: 45
        if "total trades" in line_lower or "trades:" in line_lower:
            try:
                for word in line.split():
                    if word.isdigit():
                        results["total_trades"] = int(word)
                        break
            except ValueError:
                pass
        
        # Win Rate: 65.5%
        if "win rate" in line_lower or "winrate" in line_lower:
            try:
                for part in line.replace('%', '').split():
                    try:
                        val = float(part)
                        if 0 <= val <= 100:
                            results["win_rate"] = val
                            break
                    except ValueError:
                        continue
            except ValueError:
                pass
        
        # Max Drawdown: -5.2%
        if "drawdown" in line_lower:
            try:
                for part in line.replace('%', '').split():
                    try:
                        val = float(part)
                        if -100 <= val <= 0:
                            results["max_drawdown"] = val
                            break
                    except ValueError:
                        continue
            except ValueError:
                pass
        
        # PnL: $234.56 or PnL: 2.34%
        if "pnl" in line_lower or "p&l" in line_lower:
            try:
                if '$' in line:
                    parts = line.split('$')
                    if len(parts) > 1:
                        results["pnl"] = float(parts[1].strip().replace(',', ''))
                else:
                    for part in line.replace('%', '').split():
                        try:
                            val = float(part)
                            results["pnl"] = val
                            break
                        except ValueError:
                            continue
            except ValueError:
                pass
    
    return results

def run_backtest(symbol: str, config_path: Path, days: int) -> dict:
    """Execute backtest and return parsed results"""
    env = os.environ.copy()
    env["REGIME_CONFIG"] = str(config_path)
    
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        "--symbol", symbol,
        "--days", str(days)
    ]
    
    print(f"      🔴 Simulando {symbol} ({days} días) -> Timeout: 600s")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout (ML backtests are slow)
            env=env,
            cwd=str(PROJECT_ROOT)
        )
        output = result.stdout + result.stderr
        return parse_backtest_results(output)
    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] {symbol} - Tardó más de 10 minutos")
        return {"final_capital": 0.0, "total_trades": 0, "error": "timeout_600s"}
    except Exception as e:
        print(f"      [ERROR] {symbol}: {e}")
        return {"final_capital": 0.0, "total_trades": 0, "error": str(e)}

# ═════════════════════════════════════════════════════════════════════════════
# 5. TOURNAMENT EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def run_tournament(symbols: list, days: int) -> dict:
    """Run the full grid search tournament"""
    
    print("\n" + "=" * 70)
    print("  🥷 NINJA SYSTEM v4.0 - SYMBOL GRID SEARCH TOURNAMENT")
    print("=" * 70)
    print(f"  Símbolos: {len(symbols)}")
    print(f"  Escenarios: {list(SCENARIOS.keys())}")
    print(f"  Días de backtest: {days}")
    print("=" * 70 + "\n")
    
    # Load base configuration
    if not BASE_CONFIG_FILE.exists():
        print(f"[ERROR] Config file not found: {BASE_CONFIG_FILE}")
        sys.exit(1)
    
    base_config = load_yaml(BASE_CONFIG_FILE)
    
    # Results storage
    all_results = {}
    winners = {}
    
    # Run tournament
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 🎯 SYMBOL: {symbol}")
        print("-" * 50)
        
        symbol_results = {}
        
        for scenario_name, scenario_overrides in SCENARIOS.items():
            print(f"  ⚔️  Escenario: {scenario_name}")
            
            # Apply scenario to config
            test_config = apply_scenario_to_config(base_config, symbol, scenario_overrides)
            
            # Save temp config
            save_yaml(TEMP_CONFIG_FILE, test_config)
            
            # Run backtest
            results = run_backtest(symbol, TEMP_CONFIG_FILE, days)
            symbol_results[scenario_name] = results
            
            capital = results.get("final_capital", 0)
            trades = results.get("total_trades", 0)
            print(f"      → Capital: ${capital:,.2f} | Trades: {trades}")
        
        # Determine winner for this symbol
        winner = max(symbol_results.items(), key=lambda x: x[1].get("final_capital", 0))
        winners[symbol] = {
            "scenario": winner[0],
            "capital": winner[1].get("final_capital", 0),
            "trades": winner[1].get("total_trades", 0),
            "win_rate": winner[1].get("win_rate", 0)
        }
        
        all_results[symbol] = symbol_results
        print(f"  🏆 GANADOR: {winner[0]} (${winner[1].get('final_capital', 0):,.2f})")
    
    # Cleanup temp file
    if TEMP_CONFIG_FILE.exists():
        TEMP_CONFIG_FILE.unlink()
    
    return {"all_results": all_results, "winners": winners}

def generate_report(results: dict, days: int) -> str:
    """Generate final report with YAML output for winners"""
    
    winners = results["winners"]
    
    report_lines = [
        "",
        "=" * 70,
        "  📊 REPORTE FINAL - SYMBOL GRID SEARCH",
        "=" * 70,
        f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Días de backtest: {days}",
        "",
        "  GANADORES POR SÍMBOLO:",
        "-" * 70,
    ]
    
    for symbol, data in winners.items():
        scenario = data["scenario"]
        capital = data["capital"]
        trades = data["trades"]
        report_lines.append(f"  {symbol:15} → {scenario:12} | Capital: ${capital:>10,.2f} | Trades: {trades:>4}")
    
    report_lines.extend([
        "",
        "-" * 70,
        "  YAML PARA regime_config.live.yaml:",
        "-" * 70,
        "",
        "SYMBOL_OVERRIDES:"
    ])
    
    # Generate YAML for winners (only non-DEFAULT)
    for symbol, data in winners.items():
        scenario = data["scenario"]
        if scenario != "DEFAULT" and scenario in SCENARIOS:
            overrides = SCENARIOS[scenario]
            if overrides:
                report_lines.append(f"  {symbol}:")
                for regime, params in overrides.items():
                    params_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                    report_lines.append(f"    {regime}: {{ {params_str} }}")
    
    report_lines.extend([
        "",
        "=" * 70,
        "  Copia el bloque SYMBOL_OVERRIDES arriba a tu regime_config.live.yaml",
        "=" * 70,
        ""
    ])
    
    return "\n".join(report_lines)

# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ninja Symbol Grid Search Tournament")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of symbols (default: all)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days for backtest (default: 7)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_SYMBOLS
    
    # Run tournament
    results = run_tournament(symbols, args.days)
    
    # Generate and print report
    report = generate_report(results, args.days)
    print(report)
    
    # Save JSON results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Resultados JSON guardados en: {output_path}")
    
    # Save default report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Reporte guardado en: {report_file}")

if __name__ == "__main__":
    main()
