import numpy as np
import time
import logging
from datetime import datetime
from collections import deque
import threading

from data_manager import DataManager

logger = logging.getLogger(__name__)

class OrderbookProcessor:
    """
    Processes orderbook data and calculates trading cost metrics.
    Manages data flow between WebSocket and models.
    """
    
    def __init__(self, slippage_model, fee_model, impact_model, maker_taker_model, max_history=1000):
        """
        Initialize the orderbook processor with required models.
        
        Args:
            slippage_model: Model for calculating slippage
            fee_model: Model for calculating fees
            impact_model: Model for calculating market impact
            maker_taker_model: Model for predicting maker/taker proportion
            max_history (int): Maximum number of historical data points to keep
        """
        self.slippage_model = slippage_model
        self.fee_model = fee_model
        self.impact_model = impact_model
        self.maker_taker_model = maker_taker_model
        
        # Initialize data manager for database operations
        self.data_manager = DataManager()
        
        # Latest orderbook data
        self.latest_orderbook = None
        
        # Latest calculated metrics
        self.latest_metrics = {}
        
        # Performance metrics
        self.processing_times = deque(maxlen=max_history)
        self.start_time = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def process_orderbook(self, orderbook_data, quantity):
        """
        Process incoming orderbook data and calculate metrics.
        
        Args:
            orderbook_data (dict): L2 orderbook data from WebSocket
            quantity (float): Trade size in USD
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Store the latest orderbook
                self.latest_orderbook = orderbook_data
                
                # Prepare orderbook data for models
                if 'asks' in orderbook_data and 'bids' in orderbook_data:
                    asks = np.array(orderbook_data['asks'], dtype=float)
                    bids = np.array(orderbook_data['bids'], dtype=float)
                    
                    # Skip processing if orderbook is empty
                    if len(asks) == 0 or len(bids) == 0:
                        logger.warning("Empty orderbook received, skipping processing")
                        return
                    
                    # Extract best prices
                    best_ask = float(asks[0][0])
                    best_bid = float(bids[0][0])
                    mid_price = (best_ask + best_bid) / 2
                    
                    # Calculate base quantity (in crypto units)
                    base_quantity = quantity / mid_price
                    
                    # Calculate metrics
                    slippage = self.slippage_model.calculate(asks, bids, base_quantity, True)  # True for buy, False for sell
                    
                    # Predict maker/taker proportion
                    maker_proportion, taker_proportion = self.maker_taker_model.predict(
                        asks, bids, base_quantity, orderbook_data.get('symbol', '')
                    )
                    
                    # Calculate expected fees
                    fees = self.fee_model.calculate(quantity, maker_proportion, taker_proportion)
                    
                    # Calculate market impact
                    market_impact = self.impact_model.calculate(
                        asks, bids, base_quantity, mid_price, orderbook_data.get('symbol', '')
                    )
                    
                    # Calculate net cost
                    net_cost = slippage + fees + market_impact
                    
                    # Store the calculated metrics
                    elapsed_time = time.time() - start_time
                    self.processing_times.append(elapsed_time)
                    
                    self.latest_metrics = {
                        'timestamp': orderbook_data.get('timestamp', datetime.now().isoformat()),
                        'slippage': slippage,
                        'fees': fees,
                        'market_impact': market_impact,
                        'net_cost': net_cost,
                        'maker_proportion': maker_proportion,
                        'taker_proportion': taker_proportion,
                        'processing_latency': elapsed_time,
                        'mid_price': mid_price
                    }
                    
                    # Save data to database (every 5 updates to avoid overwhelming the DB)
                    if len(self.processing_times) % 5 == 0:
                        # Save orderbook snapshot
                        self.data_manager.save_orderbook_snapshot(orderbook_data)
                        
                        # Save performance metric
                        self.data_manager.save_performance_metric('processing_latency', elapsed_time)
                        
                        # Only save simulation result occasionally to avoid DB flooding
                        if len(self.processing_times) % 20 == 0:
                            # Prepare simulation parameters
                            simulation_params = {
                                'exchange': orderbook_data.get('exchange', 'unknown'),
                                'symbol': orderbook_data.get('symbol', 'unknown'),
                                'order_type': 'market',
                                'quantity': quantity,
                                'volatility': self.impact_model.volatility * 100,  # Convert back to percentage
                                'maker_fee': self.fee_model.maker_fee,
                                'taker_fee': self.fee_model.taker_fee
                            }
                            
                            # Save simulation result
                            self.data_manager.save_simulation_result(self.latest_metrics, simulation_params)
                    
                    logger.debug(f"Processed orderbook data in {elapsed_time*1000:.2f}ms")
                else:
                    logger.warning("Invalid orderbook data format")
                    
        except Exception as e:
            logger.error(f"Error processing orderbook: {str(e)}")
            
    def get_latest_orderbook(self):
        """Get the latest orderbook data."""
        with self.lock:
            return self.latest_orderbook
            
    def get_latest_metrics(self):
        """Get the latest calculated metrics."""
        with self.lock:
            return self.latest_metrics
    
    def get_latency_history(self):
        """Get processing latency history."""
        with self.lock:
            return list(self.processing_times)
    
    def get_total_processing_time(self):
        """Get total processing time since start."""
        return time.time() - self.start_time