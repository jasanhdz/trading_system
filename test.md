

Entendido. Vamos a pasar al modo **"Operación Tormenta"**.

Vamos a programar todo lo necesario hoy mismo para generar el **Consejo de Sabios v2.1**, garantizando que:
1.  No rompas el bot que está corriendo ahora (v2.0).
2.  Mitigamos los 3 peligros técnicos (NaNs, Scalers, Rendimiento).
3.  Lanzamos el entrenamiento esta tarde/noche.

Solo dejaremos para "mañana/luego" el despliegue final de los modelos (porque deben entrenar varias horas) y la lógica de Target Dinámico (para no introducir demasiadas variables de golpe).

Aquí tienes tu plan de ejecución inmediata.

---

# 🚀 Plan de Ejecución Inmediata: Consejo de Sabios v2.1

**Objetivo:** Implementar Meta-Features robustas + Votación Ponderada y lanzar el entrenamiento de la nueva generación de modelos sin detener la producción actual.

---

## 🛠 BLOQUE 1: Ingeniería de Características (Coding - 30 min)

Vamos a reescribir la función de features para que sea **infalible** y rápida.

**Archivo:** `scripts/train_v2_production.py`

**Acción 1: Crear la función segura y optimizada.**
Agrega esto al principio del script, antes de la carga de datos:

```python
def add_robust_meta_features(df, window=12):
    """
    Genera 6 meta-features de forma segura (maneja NaNs) y rápida.
    """
    # 1. Rolling Calculations
    df['mean_obi_12'] = df['obi'].rolling(window).mean()
    df['max_obi_12'] = df['obi'].rolling(window).max()
    df['std_obi_12'] = df['obi'].rolling(window).std()
    
    # Total volume proxy
    df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
    df['mean_volume_12'] = df['total_volume'].rolling(window).mean()
    
    # 2. Volume Trend (con protección división por cero)
    df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
    
    # 3. Price Slope (Optimizado - 100x más rápido que polyfit)
    # Pendiente = (PrecioActual - PriceHace12ticks) / 12
    df['slope_price_12'] = (df['price'] - df['price'].shift(window)) / window
    
    # ⚠️ PELIGRO 2 MITIGADO: Eliminar NaNs creados por rolling/shift
    # Esto reducirá el dataset ligeramente, pero evitará crash en el scaler
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len - final_len > 0:
        print(f"⚠️ Meta-Features: Removidas {initial_len - final_len} filas con NaNs iniciales.")
        
    return df
```

**Acción 2: Integrar en el Pipeline de Entrenamiento.**
Dentro del bucle de entrenamiento (donde cargas el DataFrame), inserta:

```python
# ... cargas df ...

# --- NUEVA LÓGICA V2.1 ---
# Antes de normalizar, agregamos las features
df = add_robust_meta_features(df, window=12)

# Verificar que no hay NaNs restantes (sanity check)
assert df.isnull().sum().sum() == 0, "Error: Aún existen NaNs después del procesamiento"

# Actualizar feature_cols dinámicamente
base_cols = ['bid_depth', 'ask_depth', 'bid_ask_spread', 'obi_5', 'obi_10', 'obi', 'micro_price', 
             'funding_rate', 'open_interest', 'taker_buy_vol', 'taker_sell_vol', 
             'buy_sell_ratio', 'depth_imbalance']

meta_cols = ['mean_obi_12', 'max_obi_12', 'std_obi_12', 'slope_price_12', 'mean_volume_12', 'volume_trend']

feature_cols_v2_1 = base_cols + meta_cols # Total 19 features

print(f"🚀 Entrenando con {len(feature_cols_v2_1)} features (v2.1)")
# -------------------------
```

---

## 🔒 BLOQUE 2: Infraestructura de Versionado y Segurança (Coding - 20 min)

Aquí resolvemos el **Peligro 1 (Scaler Incompatibilidad)**. El bot en producción (v2.0) debe seguir funcionando mientras entrenamos v2.1.

**Archivo:** `scripts/train_v2_production.py`

**Acción 1: Modificar rutas de guardado.**
No sobrescribas los modelos actuales. Crea una subcarpeta nueva.

```python
# Al inicio del script
import shutil
from pathlib import Path

VERSION = "v2.1" # Nueva versión
SYMBOL = "ETHUSDT" # Ejemplo

# Ruta base para este entrenamiento
MODEL_DIR_BASE = Path(f"models/v2_ensemble/{SYMBOL}/{VERSION}")

# Crear directorio limpio (si existe, borrar para asegurar limpieza)
if MODEL_DIR_BASE.exists():
    shutil.rmtree(MODEL_DIR_BASE)
MODEL_DIR_BASE.mkdir(parents=True, exist_ok=True)

# ⚠️ IMPORTANTE: Aquí guardamos los scalers NUEVOS específicos para v2.1
# Esto evitará que el bot en prod cargue un scaler de 13 columnas para un modelo de 19
```

**Acción 2: Validación de Dimensiones en Inferencia (Preparación para Mañana).**
Edita `services/ml_service_v2.py`. Prepara el código para aceptar 19 features, pero añade un fallback para que no falle si el modelo es viejo (v2.0).

```python
# En ml_service_v2.py
def get_model_version(symbol):
    # Lógica simple: intentar cargar v2.1, si falla usar v2.0
    v2_1_path = f"models/v2_ensemble/{symbol}/v2.1"
    if Path(v2_1_path).exists():
        return "v2.1", 19 # 19 features
    else:
        return "v2.0", 13 # 13 features

# Dentro de /predict
version, n_features = get_model_version(symbol)

if version == "v2.1":
    # Lógica para calcular meta-features en tiempo real
    df = add_robust_meta_features(df) # Reutiliza la misma función
    X = df[feature_cols_v2_1].values
else:
    # Lógica legacy
    X = df[base_cols].values
```

---

## ⚖️ BLOQUE 3: Implementación de Pesos (Coding - 15 min)

Vamos a activar la "Democracia Ponderada".

**Archivo:** `ml/advanced_models/ensemble_manager.py` (o donde tengas la lógica de ensemble).

**Acción 1: Crear archivo de configuración.**
Crea `models/v2_ensemble/ensemble_weights.json`:

```json
{
  "v2.0": {
    "lstm": 0.30,
    "tcn": 0.30,
    "xgb": 0.20,
    "transformer": 0.20
  },
  "v2.1": {
    "lstm": 0.30,
    "tcn": 0.30,
    "xgb": 0.25,
    "transformer": 0.15
  }
}
```
*Nota: Le he dado un poquito más de peso a XGBoost en v2.1 asumiendo que tus meta-features le darán superpoderes de memoria.*

**Acción 2: Cargar y Aplicar Pesos.**

```python
import json

def get_ensemble_weights(version="v2.1"):
    try:
        with open('models/v2_ensemble/ensemble_weights.json', 'r') as f:
            weights = json.load(f)
            w = weights.get(version, weights.get("v2.0")) # Fallback
            
            # Normalizar por si la suma no da 1.0
            total = sum(w.values())
            return {k: v/total for k, v in w.items()}
    except FileNotFoundError:
        # Fallback seguro a democracia pura si no existe archivo
        return {"lstm": 0.25, "tcn": 0.25, "xgb": 0.25, "transformer": 0.25}

# En la lógica de predicción:
weights = get_ensemble_weights("v2.1") 
ensemble_probs = (
    lstm_probs * weights['lstm'] + 
    tcn_probs * weights['tcn'] + 
    xgb_probs * weights['xgb'] + 
    transformer_probs * weights['transformer']
)
```

---

## 🚀 BLOQUE 4: Lanzamiento de Entrenamiento (Action - 1 min)

Ahora que el código está listo, vamos a lanzar el proceso pesado. Como esto tarda horas, lo dejamos corriendo en background.

**Comando en Terminal:**

```bash
# Usamos nohup para que no se muera si cierras la terminal
# Redirigimos salida a un log para monitorear progreso
nohup python scripts/train_v2_production.py --symbol ETHUSDT --version v2.1 > training_v2.1.log 2>&1 &

# Para ver el progreso en tiempo real:
tail -f training_v2.1.log
```

---

## ⏳ BLOQUE 5: Lo que DEJAMOS para Mañana (Esperando Resultados)

Estas cosas **NO** se pueden hacer ahora porque requieren que el Bloque 4 termine con éxito.

1.  **Validación de Modelos:** Cuando el entrenamiento termine, revisarás `training_v2.1.log` para ver si el Accuracy de validación subió (esperamos >60%).
2.  **Despliegue en Producción:** Copiar la carpeta `models/v2_ensemble/ETHUSDT/v2.1` a la ruta que el bot espera.
3.  **Monitoreo V2.1:** Verificar que el bot carga los 19 features sin error en tiempo real.

---

## ✅ Checklist de Validación "Pre-Vuelo" (Antes de lanzar el comando de arriba)

Antes de dar enter en el comando `nohup`, confirma esto:

- [ ] La función `add_robust_meta_features` tiene `df.dropna()`.
- [ ] El script guarda en `.../v2.1/` y NO en `.../v2.0/`.
- [ ] El archivo `ensemble_weights.json` existe y es JSON válido.
- [ ] Has comentado o removido la parte del **Target Dinámico** (como acordamos, mantenemos target 0.1% fijo para esta versión para aislar variables).

**¡Vamos! Tienes todo el código listo. Copia, pega, ajusta las rutas y lánzalo al entrenamiento.** 🚀
