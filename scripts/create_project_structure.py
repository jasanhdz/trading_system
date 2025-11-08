# =============================================================================
# 3. SCRIPT PARA CREAR ESTRUCTURA COMPLETA
# =============================================================================
import os
from pathlib import Path

def create_project_structure():
    """Crear toda la estructura del proyecto"""
    
    # Directorios a crear
    directories = [
        "data",
        "data/collectors", 
        "data/processors",
        "data/storage",
        "data/raw",
        "analysis",
        "analysis/features",
        "analysis/statistics", 
        "strategies",
        "backtesting",
        "utils",
        "config",
        "tests",
        "notebooks",
        "scripts",
        "logs"
    ]
    
    # Crear directorios
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
        
        # Crear __init__.py en directorios Python
        if directory not in ['logs', 'notebooks', 'scripts', 'data/raw']:
            init_file = Path(directory) / "__init__.py"
            init_file.touch()
            print(f"📄 Created: {init_file}")
    
    print("\n🎉 Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
