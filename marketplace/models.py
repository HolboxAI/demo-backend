from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MarketplaceCustomer(Base):
    __tablename__ = 'marketplace_customers'

    id = Column(String(36), primary_key=True, index=True)
    customer_identifier = Column(String(255), unique=True, index=True, nullable=False)
    product_code = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MarketplaceCustomer(id='{self.id}', customer_identifier='{self.customer_identifier}', product_code='{self.product_code}')>"
