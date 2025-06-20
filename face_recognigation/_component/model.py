# face_recognition/_component/model.py

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Create the base class for models
Base = declarative_base()

# Define the UserMetadata table
class UserMetadata(Base):
    __tablename__ = "user_metadata"

    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, index=True, unique=True)
    name = Column(String, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    timestamp = Column(DateTime)

# Database URL - replace with your actual database URL
DATABASE_URL = "postgresql://postgres:demo.holbox.ai@database-1.carkqwcosit4.us-east-1.rds.amazonaws.com:5432/face_detection"

# Create the SQLAlchemy engine and session maker
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)
