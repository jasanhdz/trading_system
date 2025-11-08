from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import settings

def get_database_engine():
    # Crear engine de base de datos con configuraciones específicas
    if "sqlite" in settings.DATABASE_URL:
        engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
    return engine

def get_session_maker():
    # Crear session maker
    engine = get_database_engine()
    return sessionmaker(bind=engine)