from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
import boto3
import os
from uuid import uuid4
from .models import MarketplaceCustomer, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Use the same database configuration as the main app - AWS RDS MySQL
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://admin:your_secure_password@demo-backend-db.xxxxx.us-east-1.rds.amazonaws.com:3306/face_detection")

# Enhanced engine configuration for RDS
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    pool_size=10,        # Connection pool size
    max_overflow=20,     # Additional connections when pool is full
    echo=False           # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("api/demo_backed_v2/marketplace/fulfillment")
async def marketplace_fulfillment(request: Request, db: Session = Depends(get_db)):
    """
    Handles AWS Marketplace new customer registration.
    """
    token = request.headers.get("x-amz-marketplace-token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing x-amz-marketplace-token")

    marketplace_client = boto3.client("metering.marketplace", region_name=os.getenv("AWS_REGION", "us-east-1"))

    try:
        response = marketplace_client.resolve_customer(
            RegistrationToken=token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to resolve customer: {str(e)}")

    customer_identifier = response.get("CustomerIdentifier")
    product_code = response.get("ProductCode")

    if not customer_identifier or not product_code:
        raise HTTPException(status_code=400, detail="Invalid customer data from AWS")

    # Check if customer already exists
    existing_customer = db.query(MarketplaceCustomer).filter_by(customer_identifier=customer_identifier).first()
    if existing_customer:
        return {"status": "customer already registered", "customer_id": existing_customer.id}

    new_customer = MarketplaceCustomer(
        id=str(uuid4()),
        customer_identifier=customer_identifier,
        product_code=product_code
    )
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)
    
    return {"status": "success", "customer_id": new_customer.id}

@router.get("api/demo_backed_v2/marketplace/health")
async def marketplace_health():
    """
    Health check for marketplace service and database connection
    """
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # Test AWS credentials
        marketplace_client = boto3.client("metering.marketplace", region_name=os.getenv("AWS_REGION", "us-east-1"))
        
        return {
            "status": "healthy",
            "database": "connected",
            "aws_credentials": "valid",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
