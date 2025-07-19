import json
import logging
import uuid
from datetime import datetime
import numpy as np
from database import get_db, OrderbookSnapshot, SimulationResult, PerformanceMetric

# Configure logging
logger = logging.getLogger(__name__)

class DataManager:
    """
    Handles data persistence and retrieval for the trade simulator.
    """
    
    def __init__(self, session_id=None):
        """
        Initialize data manager with optional session ID.
        
        Args:
            session_id (str, optional): Unique session identifier. Will generate one if not provided.
        """
        self.session_id = session_id if session_id else str(uuid.uuid4())
        logger.info(f"Data manager initialized with session ID: {self.session_id}")
        
    def save_orderbook_snapshot(self, orderbook_data):
        """
        Save a snapshot of the orderbook to the database.
        
        Args:
            orderbook_data (dict): Orderbook data with asks and bids
            
        Returns:
            int: ID of the saved snapshot or None if save failed
        """
        try:
            # Prepare data
            asks = np.array(orderbook_data.get('asks', []), dtype=float)
            bids = np.array(orderbook_data.get('bids', []), dtype=float)
            
            if len(asks) == 0 or len(bids) == 0:
                logger.warning("Empty orderbook data, not saving")
                return None
                
            # Calculate metrics
            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            mid_price = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            bid_depth = float(np.sum(bids[:, 1]))
            ask_depth = float(np.sum(asks[:, 1]))
            
            # Convert numpy arrays to lists for JSON serialization
            asks_list = asks.tolist() if isinstance(asks, np.ndarray) else asks
            bids_list = bids.tolist() if isinstance(bids, np.ndarray) else bids
            
            # Create snapshot object
            snapshot = OrderbookSnapshot(
                timestamp=datetime.now(),
                exchange=orderbook_data.get('exchange', 'unknown'),
                symbol=orderbook_data.get('symbol', 'unknown'),
                mid_price=mid_price,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                bid_levels_json=json.dumps(bids_list),
                ask_levels_json=json.dumps(asks_list)
            )
            
            # Save to database
            db = get_db()
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            
            logger.debug(f"Saved orderbook snapshot, ID: {snapshot.id}")
            return snapshot.id
            
        except Exception as e:
            logger.error(f"Error saving orderbook snapshot: {str(e)}")
            return None
            
    def save_simulation_result(self, metrics, simulation_params):
        """
        Save simulation result to the database.
        
        Args:
            metrics (dict): Calculated metrics from the simulation
            simulation_params (dict): Parameters used for the simulation
            
        Returns:
            int: ID of the saved result or None if save failed
        """
        try:
            # Create simulation result object
            result = SimulationResult(
                timestamp=datetime.now(),
                exchange=simulation_params.get('exchange', 'unknown'),
                symbol=simulation_params.get('symbol', 'unknown'),
                order_type=simulation_params.get('order_type', 'market'),
                quantity=float(simulation_params.get('quantity', 0)),
                volatility=float(simulation_params.get('volatility', 0)),
                maker_fee=float(simulation_params.get('maker_fee', 0)),
                taker_fee=float(simulation_params.get('taker_fee', 0)),
                slippage=float(metrics.get('slippage', 0)),
                fees=float(metrics.get('fees', 0)),
                market_impact=float(metrics.get('market_impact', 0)),
                net_cost=float(metrics.get('net_cost', 0)),
                maker_proportion=float(metrics.get('maker_proportion', 0)),
                taker_proportion=float(metrics.get('taker_proportion', 0)),
                mid_price=float(metrics.get('mid_price', 0)),
                execution_price=float(metrics.get('execution_price', 0)) if 'execution_price' in metrics else 0,
                processing_latency=float(metrics.get('processing_latency', 0))
            )
            
            # Save to database
            db = get_db()
            db.add(result)
            db.commit()
            db.refresh(result)
            
            logger.info(f"Saved simulation result, ID: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Error saving simulation result: {str(e)}")
            return None
            
    def save_performance_metric(self, metric_name, metric_value):
        """
        Save a performance metric to the database.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Value of the metric
            
        Returns:
            int: ID of the saved metric or None if save failed
        """
        try:
            # Create metric object
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                metric_value=float(metric_value),
                session_id=self.session_id
            )
            
            # Save to database
            db = get_db()
            db.add(metric)
            db.commit()
            db.refresh(metric)
            
            logger.debug(f"Saved performance metric: {metric_name}={metric_value}, ID: {metric.id}")
            return metric.id
            
        except Exception as e:
            logger.error(f"Error saving performance metric: {str(e)}")
            return None
            
    def get_historical_simulations(self, symbol=None, limit=100):
        """
        Get historical simulation results.
        
        Args:
            symbol (str, optional): Filter by symbol
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of simulation result objects
        """
        try:
            db = get_db()
            query = db.query(SimulationResult).order_by(SimulationResult.timestamp.desc())
            
            if symbol:
                query = query.filter(SimulationResult.symbol == symbol)
                
            results = query.limit(limit).all()
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving historical simulations: {str(e)}")
            return []
            
    def get_performance_metrics(self, metric_name=None, limit=100):
        """
        Get historical performance metrics.
        
        Args:
            metric_name (str, optional): Filter by metric name
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of performance metric objects
        """
        try:
            db = get_db()
            query = db.query(PerformanceMetric).filter(
                PerformanceMetric.session_id == self.session_id
            ).order_by(PerformanceMetric.timestamp.desc())
            
            if metric_name:
                query = query.filter(PerformanceMetric.metric_name == metric_name)
                
            metrics = query.limit(limit).all()
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving performance metrics: {str(e)}")
            return []
            
    def get_recent_orderbooks(self, symbol=None, limit=10):
        """
        Get recent orderbook snapshots.
        
        Args:
            symbol (str, optional): Filter by symbol
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of orderbook snapshot objects
        """
        try:
            db = get_db()
            query = db.query(OrderbookSnapshot).order_by(OrderbookSnapshot.timestamp.desc())
            
            if symbol:
                query = query.filter(OrderbookSnapshot.symbol == symbol)
                
            snapshots = query.limit(limit).all()
            return snapshots
            
        except Exception as e:
            logger.error(f"Error retrieving recent orderbooks: {str(e)}")
            return []