# =============================================================================
# 10. SCRIPTS/SETUP_DATABASE.PY - Script de configuración inicial
# =============================================================================
#!/usr/bin/env python3
# Script para configurar la base de datos inicial
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import db_manager
from utils.logger import setup_logger

def main():
    logger = setup_logger("setup_db")
    
    try:
        logger.info("Creating database tables...")
        db_manager.create_tables()
        logger.info("Database setup completed successfully!")
        
        # Test connection
        with db_manager.get_session() as session:
            logger.info("Database connection test: SUCCESS")
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()