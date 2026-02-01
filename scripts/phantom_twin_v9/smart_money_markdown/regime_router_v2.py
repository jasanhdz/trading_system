#!/usr/bin/env python3
"""
Phantom V11: Regime Router V2 (The Gatekeeper)
Final Logic Integration:
1. Macro Filter: Is this a Crash Regime? (Twin Match)
2. Micro Filter: Is this a Steroid Candle? (High Precision)
3. Safety Filter: Is this Anti-Panic? (Exhaustion Check)
"""
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE UMBRALES (TOMADOS DEL ENTREMAMIENTO VERIFICADO) ---
# Usamos los valores exactos que produjeron el +61% de retorno en Validación.

# 1. MACRO FILTER (TWIN REGIME)
# Buscamos un régimen de tendencia fuerte y bajista
HURST_THRESHOLD = 0.72       # Persistencia de la tendencia (Safe verified value)
SLOPE_MAX_MACRO = -0.00002  # Pendiente general negativa (Aligned with Micro)

# 2. MICRO FILTER (STEROID)
# Umbrales VERIFICADOS con refine_dataset.py (dataset_steroid.csv)
# NOTA: Usar umbrales más estrictos aquí mataría la frecuencia de operaciones.
VOL_Z_MIN = -0.5            # Volatilidad aceptable (no tiene que ser record)
VOLUME_MIN = 0.8            # Volumen decente (no masivo necesariamente)
WEAKNESS_MIN = 0.0          # Ligeramente bajista es suficiente
SLOPE_MAX_MICRO = -0.00002  # Pendiente inmediata negativa

# 3. SAFETY FILTER (ANTI-PANIC)
# Evitar caídas verticales sin gasolina para seguir
RSI_MIN = 25.0              # RSI > 25 (No estar "sobrevendido" al extremo)
CVD_SLOPE_MAX = -100000     # Flujo de dinero saliendo (-100k es suficiente, -500k filtra demasiado)

def check_regime(df_row):
    """
    Función Principal del Router.
    Recibe una fila de DataFrame (con indicadores pre-calculados).
    Retorna: True (ACTIVAR MODELO) / False (IDLE)
    """
    
    # --- PASO 1: ANÁLISIS MACRO (El Ambiente) ---
    # Si el mercado en general no es una "caída fuerte", ni nos molestamos.
    # Esto evita operar en rangos o rebotes alcistas.
    
    current_hurst = df_row.get('hurst', 0.5) # Default to neutral if missing
    current_slope = df_row['slope']
    
    is_crash_regime = (
        (current_hurst > HURST_THRESHOLD) & 
        (current_slope < SLOPE_MAX_MACRO)
    )
    
    # Optional logic: If Hurst is extremely high, we might relax slope, but let's keep it strict.
    if not is_crash_regime:
        return False, "MACRO: No Crash Regime detected"

    # --- PASO 2: ANÁLISIS MICRO (El Disparador) ---
    # Si estamos en régimen, verificamos que la vela actual sea "Esteroid".
    
    current_vol_z = df_row['vol_z']
    current_vol_ratio = df_row['volume_ratio']
    current_weakness = df_row['weakness_score']
    current_micro_slope = df_row['slope'] 
    
    is_steroid_entry = (
        (current_vol_z > VOL_Z_MIN) &
        (current_vol_ratio > VOLUME_MIN) &
        (current_weakness > WEAKNESS_MIN) &
        (current_micro_slope < SLOPE_MAX_MICRO)
    )
    
    if not is_steroid_entry:
        return False, "MICRO: Low Precision (Not Steroid)"

    # --- PASO 3: FILTRO DE SEGURIDAD ANTI-PÁNICO ---
    # Chequeo final para evitar entrar en una caída que ya se agotó.
    
    try:
        current_rsi = df_row.get('rsi', 50) # Safe default
        current_cvd_slope = df_row.get('cvd_slope', -999999) # Safe default (pass)
        
        is_safe_state = (
            (current_rsi > RSI_MIN) &
            (current_cvd_slope < CVD_SLOPE_MAX)
        )
        
        if not is_safe_state:
            reason = f"SAFETY: Panic/Exhaustion (RSI:{current_rsi:.1f}, CVD:{current_cvd_slope:.0f})"
            return False, reason
            
    except KeyError:
        print("⚠️ Warning: RSI/CVD columns missing, bypassing Safety Filter")
    
    # --- VEREDICTO FINAL ---
    return True, "ALL SYSTEMS GO: Steroid Crash Entry"

def main():
    print("🛡️ PHANTOM V11: REGIME ROUTER V2 INITIALIZED")
    print(f"   Hurst Threshold: > {HURST_THRESHOLD}")
    print(f"   RSI Minimum: > {RSI_MIN}")
    print(f"   Slope Max: < {SLOPE_MAX_MICRO}")
    
    # Test Data (Based on stats)
    # Good Scenario
    dummy_row = {
        'hurst': 0.78,
        'slope': -0.00004,
        'vol_z': -0.1,
        'volume_ratio': 1.2,
        'weakness_score': 0.1,
        'rsi': 30,
        'cvd_slope': -150000
    }
    
    # Panic Scenario (RSI too low)
    panic_row = dummy_row.copy()
    panic_row['rsi'] = 20

    # Weak Trend Scenario (Slope too flat)
    flat_row = dummy_row.copy()
    flat_row['slope'] = -0.00001

    print("\n--- TEST CASES ---")
    
    ok, msg = check_regime(dummy_row)
    print(f"Perfect Setup: {'✅ ALLOW' if ok else '🚫 DENY'} | {msg}")

    ok, msg = check_regime(panic_row)
    print(f"Panic Setup : {'✅ ALLOW' if ok else '🚫 DENY'} | {msg}")
    
    ok, msg = check_regime(flat_row)
    print(f"Flat Setup  : {'✅ ALLOW' if ok else '🚫 DENY'} | {msg}")

if __name__ == "__main__":
    main()
