import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class OrderbookSnapshot(Base):
    """Model for storing orderbook snapshots"""
    __tablename__ = "orderbook_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    exchange = Column(String, index=True)
    symbol = Column(String, index=True)
    mid_price = Column(Float)
    best_bid = Column(Float)
    best_ask = Column(Float)
    spread = Column(Float)
    bid_depth = Column(Float)  # Sum of bid quantities
    ask_depth = Column(Float)  # Sum of ask quantities
    bid_levels_json = Column(Text)  # JSON string of bid levels
    ask_levels_json = Column(Text)  # JSON string of ask levels

class SimulationResult(Base):
    """Model for storing simulation results"""
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    exchange = Column(String)
    symbol = Column(String)
    order_type = Column(String)
    quantity = Column(Float)
    volatility = Column(Float)
    maker_fee = Column(Float)
    taker_fee = Column(Float)
    slippage = Column(Float)
    fees = Column(Float)
    market_impact = Column(Float)
    net_cost = Column(Float)
    maker_proportion = Column(Float)
    taker_proportion = Column(Float)
    mid_price = Column(Float)
    execution_price = Column(Float)
    processing_latency = Column(Float)
    
class PerformanceMetric(Base):
    """Model for storing performance metrics"""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    session_id = Column(String, index=True)  # To group metrics from same session

# Create the database tables
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Initialize database tables
if __name__ == "__main__":
    create_tables()