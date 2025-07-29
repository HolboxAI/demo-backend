from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define your SQLAlchemy Base class
Base = declarative_base()

# Define the UserMetadata table
class UserMetadata(Base):
    __tablename__ = "user_metadata"

    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String(255), index=True, unique=True)  # Set length for VARCHAR
    name = Column(String(255), index=True)  # Set length for VARCHAR
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)  # Set length for VARCHAR
    timestamp = Column(DateTime)
    image_url = Column(String) 

# Database URL for MySQL (make sure to adjust as needed)
DATABASE_URL = "mysql+pymysql://nl2sql_admin:Holbox_nl2sql@nl2sql-demo.carkqwcosit4.us-east-1.rds.amazonaws.com:3306/face_detection"

# Create the SQLAlchemy engine (make sure this is before the Base.metadata.create_all call)
engine = create_engine(DATABASE_URL)

# Create the session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables in the database if they do not exist
Base.metadata.create_all(bind=engine)
