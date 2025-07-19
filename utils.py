import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def format_number(number, decimal_places=2):
    """
    Format a number with specified decimal places.
    
    Args:
        number (float): The number to format
        decimal_places (int): Number of decimal places to display
        
    Returns:
        str: Formatted number string
    """
    try:
        format_str = f"{{:.{decimal_places}f}}"
        return format_str.format(number)
    except Exception as e:
        logger.error(f"Error formatting number: {str(e)}")
        return str(number)

def calculate_latency(start_time):
    """
    Calculate elapsed time since start time.
    
    Args:
        start_time (float): Start time from time.time()
        
    Returns:
        float: Elapsed time in seconds
    """
    return time.time() - start_time

def parse_timestamp(timestamp_str):
    """
    Parse ISO format timestamp string to datetime object.
    
    Args:
        timestamp_str (str): Timestamp in ISO format
        
    Returns:
        datetime: Parsed datetime object or None if parsing fails
    """
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except Exception as e:
        logger.error(f"Error parsing timestamp: {str(e)}")
        return None

def estimate_volume_profile(orderbook, levels=10):
    """
    Estimate volume profile from orderbook data.
    
    Args:
        orderbook (dict): Orderbook data with 'asks' and 'bids' lists
        levels (int): Number of price levels to analyze
        
    Returns:
        dict: Volume profile data
    """
    try:
        if not orderbook or 'asks' not in orderbook or 'bids' not in orderbook:
            return None
            
        asks = orderbook['asks'][:levels] if len(orderbook['asks']) > levels else orderbook['asks']
        bids = orderbook['bids'][:levels] if len(orderbook['bids']) > levels else orderbook['bids']
        
        ask_volume = sum(float(level[1]) for level in asks)
        bid_volume = sum(float(level[1]) for level in bids)
        total_volume = ask_volume + bid_volume
        
        # Avoid division by zero
        if total_volume == 0:
            return {
                'ask_percentage': 50,
                'bid_percentage': 50,
                'imbalance': 0
            }
            
        ask_percentage = (ask_volume / total_volume) * 100
        bid_percentage = (bid_volume / total_volume) * 100
        imbalance = (bid_volume - ask_volume) / total_volume
        
        return {
            'ask_percentage': ask_percentage,
            'bid_percentage': bid_percentage,
            'imbalance': imbalance
        }
        
    except Exception as e:
        logger.error(f"Error estimating volume profile: {str(e)}")
        return None

def calculate_vwap(orderbook_side, quantity):
    """
    Calculate Volume Weighted Average Price for a given quantity.
    
    Args:
        orderbook_side (list): List of [price, quantity] for one side of the orderbook
        quantity (float): Quantity to fill
        
    Returns:
        float: VWAP price or None if calculation fails
    """
    try:
        remaining_qty = quantity
        total_cost = 0
        filled_qty = 0
        
        for price, size in orderbook_side:
            price = float(price)
            size = float(size)
            
            if remaining_qty <= 0:
                break
                
            executed = min(remaining_qty, size)
            total_cost += executed * price
            filled_qty += executed
            remaining_qty -= executed
        
        # If orderbook depth is insufficient
        if filled_qty < quantity:
            logger.warning(f"Orderbook depth insufficient for quantity {quantity}, filled only {filled_qty}")
            if filled_qty == 0:
                return None
        
        return total_cost / filled_qty
        
    except Exception as e:
        logger.error(f"Error calculating VWAP: {str(e)}")
        return None