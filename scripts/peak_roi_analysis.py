#!/usr/bin/env python3
"""
Peak ROI Analysis - Analiza los logs del bot para encontrar el Peak ROI
positivo y negativo de cada operación perdedora.
"""
import re
from pathlib import Path
from collections import defaultdict

LOG_FILE = Path.home() / ".pm2/logs/01-Trading-Bot-out.log"

# Regex para limpiar códigos ANSI
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def parse_log_line(line):
    """Extrae datos de una línea de log del bot."""
    # Limpiar códigos ANSI primero
    clean_line = ANSI_ESCAPE.sub('', line)
    
    # Buscar patrón: TIME | SYMBOL | SIDE | ENTRY | MARK | ROI%
    # Ejemplo limpio: │ 7:49:21 AM │ DOGEUSDT │ SHORT │ 0.151560 │ 0.151480 │ +0.17%
    pattern = r'(\d+:\d+:\d+\s+[AP]M)\s+.*?(\w+USDT)\s+.*?(LONG|SHORT)\s+.*?([\d.]+)\s+.*?([\d.]+)\s+.*?([+-]?[\d.]+)%'
    match = re.search(pattern, clean_line)
    if match:
        return {
            "time": match.group(1),
            "symbol": match.group(2),
            "side": match.group(3),
            "entry": float(match.group(4)),
            "mark": float(match.group(5)),
            "roi": float(match.group(6))
        }
    return None

def analyze_peak_roi():
    """Analiza el log completo y extrae Peak ROI por símbolo/lado."""
    print("📊 Analizando Peak ROI de operaciones perdedoras...")
    print("="*80)
    
    # Leer log
    with open(LOG_FILE, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    
    lines = content.split('\n')
    print(f"Total líneas en log: {len(lines):,}")
    
    # Agrupar ROI por símbolo y lado
    roi_history = defaultdict(list)
    
    parsed_count = 0
    for line in lines:
        data = parse_log_line(line)
        if data:
            parsed_count += 1
            key = (data['symbol'], data['side'])
            roi_history[key].append(data['roi'])
    
    print(f"Líneas parseadas: {parsed_count:,}")
    print(f"Combinaciones símbolo/lado: {len(roi_history)}")
    
    # Mostrar Peak ROI por combinación
    print("\n📉 PEAK ROI POR SÍMBOLO/LADO (Historial Completo)")
    print("="*80)
    print(f"{'Par':<12} {'Lado':<7} {'Peak+':<12} {'Peak-':<12} {'Samples':<10} {'Stop@3%?'}")
    print("-"*80)
    
    total_salvable = 0
    total_positions = 0
    
    for (symbol, side), rois in sorted(roi_history.items()):
        peak_pos = max(rois)
        peak_neg = min(rois)
        count = len(rois)
        salvable = peak_pos >= 3.0
        if salvable:
            total_salvable += 1
        total_positions += 1
        
        flag = "✅ SÍ" if salvable else "❌ NO"
        print(f"{symbol:<12} {side:<7} +{peak_pos:>6.2f}%     {peak_neg:>6.2f}%     {count:<10} {flag}")
    
    print("="*80)
    print(f"\n📈 RESUMEN:")
    print(f"   Posiciones que alcanzaron +3% ROI: {total_salvable}/{total_positions}")
    print(f"   → Con trailing stop al 3%, estas posiciones hubieran cerrado en ganancia.")

if __name__ == "__main__":
    analyze_peak_roi()
