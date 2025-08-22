from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
import boto3
import os
from uuid import uuid4
from .models import MarketplaceCustomer, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import text


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
USAGE_PLAN_ID = os.getenv("USAGE_PLAN_ID","xd5iuh")
# Use the same database configuration as the main app - AWS RDS MySQL
DATABASE_URL = os.getenv("DATABASE_URL")

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

def mkp_client():
    return boto3.client("meteringmarketplace", region_name=AWS_REGION)

def apigw_client():
    return boto3.client("apigateway", region_name=AWS_REGION)

def provision_api_key_for_customer(customer_identifier: str) -> tuple[str, str]:
    """
    Creates an API key and attaches it to the configured Usage Plan.
    Returns: (api_key_id, api_key_value)
    """
    apigw = apigw_client()

    create_resp = apigw.create_api_key(
        name=f"mkp-{customer_identifier}",
        enabled=True,
        generateDistinctId=True
    )
    api_key_id = create_resp["id"]

    get_key = apigw.get_api_key(apiKey=api_key_id, includeValue=True)
    api_key_value = get_key.get("value")

    apigw.create_usage_plan_key(
        usagePlanId=USAGE_PLAN_ID,
        keyId=api_key_id,
        keyType="API_KEY"
    )

    return api_key_id, api_key_value

@router.post("/api/demo_backend_v2/marketplace/fulfillment")
async def marketplace_fulfillment(request: Request, db: Session = Depends(get_db)):
    """
    Handles AWS Marketplace new customer registration.
    """

    form_fields = await request.form()
    token = form_fields.get("x-amzn-marketplace-token")

    if not token:
        raise HTTPException(status_code=400, detail="Missing x-amzn-marketplace-token")

    marketplace_client = mkp_client()

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
        return {"status": "customer already registered",
                "customer_id": existing_customer.id,
                "api_key": existing_customer.api_key_value
                }

    new_customer = MarketplaceCustomer(
        id=str(uuid4()),
        customer_identifier=customer_identifier,
        product_code=product_code
    )
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)

    try:
        api_key_id, api_key_value = provision_api_key_for_customer(customer_identifier)
    except Exception as e:
        # Clean up DB record if provisioning fails
        db.delete(new_customer)
        db.commit()
        raise HTTPException(status_code=500, detail=f"API Gateway provisioning failed: {str(e)}")

    # Save the API key mapping
    new_customer.api_key_id = api_key_id
    new_customer.api_key_value = api_key_value
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)
    
    return {
        "status": "success",
        "customer_id": new_customer.id,
        "api_key": new_customer.api_key_value
    }

@router.get("api/demo_backed_v2/marketplace/health")
async def marketplace_health():
    """
    Health check for marketplace service and database connection
    """
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    try:
        # Test AWS credentials
        marketplace_client = boto3.client("meteringmarketplace", region_name=os.getenv("AWS_REGION", "us-east-1"))
        aws_status = "valid"
    except Exception as e:
        aws_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "aws_credentials": aws_status,
        "timestamp": datetime.utcnow().isoformat()
    }
