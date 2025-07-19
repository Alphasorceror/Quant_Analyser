import websocket
import logging
import json
import time
import threading

logger = logging.getLogger(__name__)

class WebSocketClient:
    """
    WebSocket client for connecting to cryptocurrency exchange APIs.
    Handles connection, message reception, and error handling.
    """
    
    def __init__(self, url, on_message=None, on_error=None, on_close=None, on_open=None):
        """
        Initialize the WebSocket client.
        
        Args:
            url (str): WebSocket endpoint URL
            on_message (callable, optional): Callback for message events
            on_error (callable, optional): Callback for error events
            on_close (callable, optional): Callback for connection close events
            on_open (callable, optional): Callback for connection open events
        """
        self.url = url
        self.ws = None
        self.connected = False
        self.reconnect_interval = 5  # seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Set custom callbacks
        self.custom_on_message = on_message
        self.custom_on_error = on_error
        self.custom_on_close = on_close
        self.custom_on_open = on_open
        
    def on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        
        Args:
            ws (WebSocketApp): WebSocket instance
            message (str): Received message
        """
        try:
            # Call custom on_message callback if provided
            if self.custom_on_message:
                self.custom_on_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def on_error(self, ws, error):
        """
        Handle WebSocket errors.
        
        Args:
            ws (WebSocketApp): WebSocket instance
            error (Exception): Error that occurred
        """
        logger.error(f"WebSocket error: {str(error)}")
        
        # Call custom on_error callback if provided
        if self.custom_on_error:
            self.custom_on_error(error)
            
    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket connection closure.
        
        Args:
            ws (WebSocketApp): WebSocket instance
            close_status_code (int): Status code of closure
            close_msg (str): Closure message
        """
        logger.info(f"WebSocket connection closed: {close_status_code}, {close_msg}")
        self.connected = False
        
        # Call custom on_close callback if provided
        if self.custom_on_close:
            self.custom_on_close(close_status_code, close_msg)
            
        # Attempt to reconnect if not a manual close
        if close_status_code != 1000:  # 1000 is normal closure
            self._reconnect()
            
    def on_open(self, ws):
        """
        Handle WebSocket connection opening.
        
        Args:
            ws (WebSocketApp): WebSocket instance
        """
        logger.info(f"WebSocket connection established to {self.url}")
        self.connected = True
        self.reconnect_attempts = 0
        
        # Call custom on_open callback if provided
        if self.custom_on_open:
            self.custom_on_open()
    
    def connect(self):
        """Establish connection to the WebSocket endpoint."""
        try:
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket connection (this is a blocking call)
            self.ws.run_forever()
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            self.connected = False
    
    def disconnect(self):
        """Close the WebSocket connection."""
        if self.ws and self.connected:
            logger.info("Closing WebSocket connection...")
            self.ws.close()
            self.connected = False
    
    def _reconnect(self):
        """Attempt to reconnect to the WebSocket."""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            
            # Wait before reconnecting
            time.sleep(self.reconnect_interval)
            
            # Reconnect in a new thread
            threading.Thread(target=self.connect, daemon=True).start()
        else:
            logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached.")