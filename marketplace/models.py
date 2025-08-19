from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MarketplaceCustomer(Base):
    __tablename__ = 'marketplace_customers'

    id = Column(String(36), primary_key=True, index=True)
    customer_identifier = Column(String(255), unique=True, index=True, nullable=False)
    product_code = Column(String(255), nullable=False)
    api_key_id = Column(String(64), unique=True, index=True, nullable=True)
    api_key_value = Column(String(512), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return (
            f"<MarketplaceCustomer(id='{self.id}', "
            f"customer_identifier='{self.customer_identifier}', "
            f"product_code='{self.product_code}', "
            f"api_key_id='{self.api_key_id}')>"
        )
