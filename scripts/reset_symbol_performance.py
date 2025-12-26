import json
import os

FILE_PATH = "binance-futures-bot-ts/data/symbol_performance.json"

def reset_performance():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    print("Estado actual:")
    print(f"Bloqueados: {data.get('blocked', [])}")
    
    # 1. Limpiar lista de bloqueados
    data['blocked'] = []
    
    # 2. Resetear contadores de pérdidas/ganancias y limpiar historial
    # Opcional: ¿Queremos borrar TODO el historial o solo desbloquear?
    # El usuario dijo "desbloquee y limpies el file", así que asumo un reset completo para empezar fresco.
    
    data['winners'] = []
    data['performance'] = {} 
    
    # Si queremos mantener la estructura pero vacía:
    # for symbol in data['performance']:
    #     data['performance'][symbol]['wins'] = 0
    #     data['performance'][symbol]['losses'] = 0
    #     data['performance'][symbol]['history'] = []
    
    # Pero "limpiar el file" sugiere dejarlo como nuevo.
    
    print("\nAplicando cambios...")
    print(f"Bloqueados (Nuevo): {data['blocked']}")
    
    with open(FILE_PATH, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("✅ Archivo reseteado exitosamente.")

if __name__ == "__main__":
    reset_performance()
