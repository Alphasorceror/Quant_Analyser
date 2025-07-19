import numpy as np
from scipy import stats
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class SlippageModel:
    """
    Model for estimating slippage using linear or quantile regression.
    Slippage is the difference between expected execution price and mid price.
    """
    
    def __init__(self):
        """Initialize the slippage model."""
        self.history = []  # To store historical data for model training
        
    def calculate(self, asks, bids, quantity, is_buy):
        """
        Calculate expected slippage for a given order.
        
        Args:
            asks (numpy.ndarray): Ask orders in format [[price, quantity], ...]
            bids (numpy.ndarray): Bid orders in format [[price, quantity], ...]
            quantity (float): Order quantity in base currency
            is_buy (bool): True for buy orders, False for sell orders
            
        Returns:
            float: Expected slippage as a percentage of the mid price
        """
        try:
            # Get best prices
            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            mid_price = (best_ask + best_bid) / 2
            
            # Determine the side of the book to work with
            book_side = asks if is_buy else bids
            
            # Calculate the weighted average price for the given quantity
            remaining_qty = quantity
            total_cost = 0
            
            for price, size in book_side:
                price = float(price)
                size = float(size)
                
                if remaining_qty <= 0:
                    break
                    
                filled_qty = min(remaining_qty, size)
                total_cost += filled_qty * price
                remaining_qty -= filled_qty
            
            # If orderbook depth is not enough, use the last price for remaining quantity
            if remaining_qty > 0:
                total_cost += remaining_qty * float(book_side[-1][0])
                
            # Calculate effective price
            effective_price = total_cost / quantity
            
            # Calculate slippage as a percentage of mid price
            if is_buy:
                slippage = (effective_price - mid_price) / mid_price
            else:
                slippage = (mid_price - effective_price) / mid_price
                
            # Keep slippage positive for consistency
            slippage = abs(slippage)
            
            # Store result for historical analysis
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'mid_price': mid_price,
                'effective_price': effective_price,
                'quantity': quantity,
                'slippage': slippage
            })
            
            # Limit history size
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
                
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {str(e)}")
            return 0.001  # Default slippage of 0.1% if calculation fails


class FeeModel:
    """
    Model for calculating trading fees based on exchange fee structure
    and predicted maker/taker proportion.
    """
    
    def __init__(self, maker_fee, taker_fee):
        """
        Initialize the fee model with exchange-specific fee rates.
        
        Args:
            maker_fee (float): Maker fee rate as a decimal (e.g., 0.0008 for 0.08%)
            taker_fee (float): Taker fee rate as a decimal (e.g., 0.0008 for 0.08%)
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
    def calculate(self, notional_value, maker_proportion, taker_proportion):
        """
        Calculate expected fees for a trade.
        
        Args:
            notional_value (float): The total value of the trade in USD
            maker_proportion (float): Proportion of order expected to be executed as maker
            taker_proportion (float): Proportion of order expected to be executed as taker
            
        Returns:
            float: Expected fee as a percentage of the notional value
        """
        try:
            # Calculate weighted fee based on maker/taker proportions
            weighted_fee = (maker_proportion * self.maker_fee) + (taker_proportion * self.taker_fee)
            
            # Return fee as percentage of notional value
            return weighted_fee
            
        except Exception as e:
            logger.error(f"Error calculating fees: {str(e)}")
            return self.taker_fee  # Default to taker fee if calculation fails


class AlmgrenChrissModel:
    """
    Implementation of the Almgren-Chriss market impact model.
    
    This model estimates market impact using orderbook data and volatility.
    """
    
    def __init__(self, volatility, permanence_factor=0.1, temporary_factor=0.5):
        """
        Initialize the Almgren-Chriss model.
        
        Args:
            volatility (float): Market volatility as a decimal
            permanence_factor (float): Factor for permanent impact component
            temporary_factor (float): Factor for temporary impact component
        """
        self.volatility = volatility
        self.permanence_factor = permanence_factor
        self.temporary_factor = temporary_factor
        
    def calculate(self, asks, bids, quantity, mid_price, symbol):
        """
        Calculate expected market impact for a given order.
        
        Args:
            asks (numpy.ndarray): Ask orders in format [[price, quantity], ...]
            bids (numpy.ndarray): Bid orders in format [[price, quantity], ...]
            quantity (float): Order quantity in base currency
            mid_price (float): Current mid price
            symbol (str): Trading pair symbol
            
        Returns:
            float: Expected market impact as a percentage of the mid price
        """
        try:
            # Calculate average daily volume (ADV) proxy from orderbook
            # This is a simplification; in production, historical data would be used
            total_volume = np.sum(asks[:, 1]) + np.sum(bids[:, 1])
            adv_proxy = total_volume * 24 * 3600 / 60  # Assuming orderbook represents ~1 minute of trading
            
            # Calculate market depth as the sum of quantities within 0.5% of mid price
            price_threshold = mid_price * 0.005
            
            depth_asks = np.sum(asks[asks[:, 0] <= mid_price + price_threshold, 1])
            depth_bids = np.sum(bids[bids[:, 0] >= mid_price - price_threshold, 1])
            market_depth = depth_asks + depth_bids
            
            # Estimate spread cost (temporary impact)
            spread = float(asks[0][0]) - float(bids[0][0])
            spread_cost = spread / mid_price
            
            # Calculate participation rate (order size relative to market depth)
            participation_rate = min(quantity / (market_depth + 1e-10), 0.3)  # Cap at 30%
            
            # Calculate permanent impact using Almgren-Chriss formula
            # Permanent impact is proportional to volatility and square root of participation rate
            permanent_impact = self.permanence_factor * self.volatility * np.sqrt(participation_rate)
            
            # Calculate temporary impact
            # Temporary impact is proportional to spread and participation rate
            temporary_impact = self.temporary_factor * spread_cost * np.power(participation_rate, 0.6)
            
            # Total market impact
            total_impact = permanent_impact + temporary_impact
            
            # Log calculation details for debugging
            logger.debug(f"Market impact calculation for {symbol}:")
            logger.debug(f"  Volatility: {self.volatility:.6f}")
            logger.debug(f"  Spread cost: {spread_cost:.6f}")
            logger.debug(f"  Participation rate: {participation_rate:.6f}")
            logger.debug(f"  Permanent impact: {permanent_impact:.6f}")
            logger.debug(f"  Temporary impact: {temporary_impact:.6f}")
            logger.debug(f"  Total impact: {total_impact:.6f}")
            
            return total_impact
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            return 0.0005  # Default market impact of 0.05% if calculation fails


class MakerTakerModel:
    """
    Model for predicting the proportion of an order that will be executed 
    as maker versus taker orders using logistic regression.
    """
    
    def __init__(self):
        """Initialize the maker/taker prediction model."""
        # We'd normally train this model with historical data
        # For now, we'll use a simple heuristic
        pass
        
    def predict(self, asks, bids, quantity, symbol):
        """
        Predict the maker/taker proportion for a given order.
        
        Args:
            asks (numpy.ndarray): Ask orders in format [[price, quantity], ...]
            bids (numpy.ndarray): Bid orders in format [[price, quantity], ...]
            quantity (float): Order quantity in base currency
            symbol (str): Trading pair symbol
            
        Returns:
            tuple: (maker_proportion, taker_proportion)
        """
        try:
            # Calculate orderbook imbalance (bid volume / total volume)
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            total_volume = bid_volume + ask_volume
            
            # Avoid division by zero
            if total_volume == 0:
                return 0.2, 0.8  # Default to 20% maker, 80% taker
                
            bid_ask_imbalance = bid_volume / total_volume
            
            # Calculate spread as percentage of mid price
            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            mid_price = (best_ask + best_bid) / 2
            spread_pct = (best_ask - best_bid) / mid_price
            
            # Calculate order size relative to market depth
            market_depth = bid_volume + ask_volume
            relative_size = min(quantity / market_depth, 1.0)
            
            # Logistic function to estimate maker proportion
            # Parameters are chosen to reflect typical market behavior
            # Higher spread → more likely to be maker
            # Larger order size → less likely to be maker
            # Higher bid/ask imbalance → more likely to be maker if selling, less if buying
            
            # For a buy order (assuming we're implementing buy):
            # Favorable conditions for maker: High asks relative to bids (low imbalance),
            # high spread, small order size
            
            # Logistic regression factors
            spread_factor = 10.0  # Higher spread increases maker proportion
            size_factor = -2.0    # Larger size decreases maker proportion
            imbalance_factor = -1.0  # Higher bid volume (for buys) decreases maker proportion
            
            # Logistic regression for maker proportion
            logit = 0.0 + \
                   (spread_factor * spread_pct) + \
                   (size_factor * relative_size) + \
                   (imbalance_factor * (bid_ask_imbalance - 0.5))
                   
            # Apply logistic function
            maker_proportion = 1.0 / (1.0 + np.exp(-logit))
            
            # Ensure reasonable bounds
            maker_proportion = max(0.05, min(0.8, maker_proportion))
            taker_proportion = 1.0 - maker_proportion
            
            return maker_proportion, taker_proportion
            
        except Exception as e:
            logger.error(f"Error predicting maker/taker proportion: {str(e)}")
            return 0.2, 0.8  # Default to 20% maker, 80% taker if prediction fails