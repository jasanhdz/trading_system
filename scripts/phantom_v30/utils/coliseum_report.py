#!/usr/bin/env python3
import re
from pathlib import Path

def print_coliseum_history(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Busca bloques de COLISEUM
    matches = re.finditer(r'🏟️ COLISEUM: Evaluating all fighters on CPU...[\s\S]*?(?:🚀 PROMOTION!|🛡️ DEFENSE!|⚠️ No Champion & Challenger rejected|💀 BLOCKED!|✅ First Champion crowned!)', content)
    
    print("\n--- 🏟️ HISTORIAL DEL COLISEO (V31 Lineage) ---")
    
    count = 0
    for match in matches:
        block = match.group(0)
        
        # Filtramos la parte donde sale -inf para mostrar 'V30 Incompatible' en cambio
        block = block.replace('🏆 Champion:       $-inf (P95 DD: 0.0%)', '🏆 Champion:       [V30 Incompatible]')
        
        # Format the block nicely
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            print(f"Batalla {count+1}:")
            for line in lines[1:]: # Skip the 'Evaluating...' line
                if "Champion: " in line or "Challenger:" in line or "PROMOTION" in line or "DEFENSE" in line or "rejected" in line or "crowned" in line:
                    print(f"  {line}")
            print("-" * 40)
            count += 1

if __name__ == '__main__':
    print_coliseum_history('/home/jasan/.pm2/logs/03-V30-Trainer-out-4.log')
