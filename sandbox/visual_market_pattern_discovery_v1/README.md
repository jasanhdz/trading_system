# VISUAL_MARKET_PATTERN_DISCOVERY_V1

Pipeline aislado, causal y exclusivamente de investigación para descubrir estados visuales recurrentes. No tiene autoridad de trading y no contiene entradas, salidas, stops, objetivos ni ejecución.

## Orden obligatorio

```bash
cd sandbox/visual_market_pattern_discovery_v1
export PYTHONPATH="$PWD/src"
../../.venv/bin/python generate_visual_dataset.py
../../.venv/bin/python compute_visual_embeddings.py
../../.venv/bin/python discover_visual_patterns.py
# En este punto cluster_assignments queda congelado y hasheado.
../../.venv/bin/python analyze_pattern_precursors.py
../../.venv/bin/python build_pattern_catalog.py
../../.venv/bin/python find_similar_frames.py --frame-id SUIUSDT_... --top-k 50
../../.venv/bin/python find_similar_frames.py --image /path/chart.png --top-k 50
```

No se permite ejecutar análisis de futuro antes de `discover_visual_patterns.py`. Los comandos posteriores verifican y preservan el hash de assignments en sus manifests.

## Encoder congelado

El entorno se inspeccionó antes de clustering: no había DINOv3, DINOv2, `timm`, `transformers` ni OpenCLIP instalados o cacheados. DINOv3 oficial requiere obtener pesos con licencia/acceso de Meta. La preregistración fija OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` como fallback reproducible; instalar con `pip install -e '.[openclip]'`. El hash efectivo de todos los tensores se guarda en `embedding_index.json`. Nunca se aceptan pesos aleatorios.

## FiftyOne opcional

El core no depende de FiftyOne. Para habilitarlo: `pip install -e '.[fiftyone]'` y llamar `vmpd_v1.fiftyone_optional.create_dataset(...)`; después abrir `fiftyone app launch VMPD_V1_M1`. La galería HTML funciona sin FiftyOne.

## Layout

Se congeló el layout exacto solicitado de cinco paneles, con SUI 15m principal. BTC 15m se deriva causalmente para métricas, pero no se añade como sexto panel para no cambiar el layout visual preregistrado. Cada panel normaliza precio y volumen sobre su propio contexto visible. Ninguna imagen contiene precio absoluto, timestamp, filename, outcome o PnL.
