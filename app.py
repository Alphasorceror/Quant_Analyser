import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import json
import plotly.graph_objects as go
from datetime import datetime
import logging

from websocket_client import WebSocketClient
from orderbook_processor import OrderbookProcessor
from models import AlmgrenChrissModel, SlippageModel, FeeModel, MakerTakerModel
from utils import format_number, calculate_latency
from history_viewer import render_history_tab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GoQuant Trade Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Cryptocurrency Trade Simulator")
st.markdown("Real-time transaction cost analysis using L2 orderbook data")

# Initialize session state if not exists
if 'orderbook_data' not in st.session_state:
    st.session_state.orderbook_data = {
        'timestamp': None,
        'exchange': None,
        'symbol': None,
        'asks': [],
        'bids': []
    }
    
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
    
if 'processor' not in st.session_state:
    st.session_state.processor = None
    
if 'connected' not in st.session_state:
    st.session_state.connected = False
    
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'processing_latency': [],
        'last_update': None
    }

# Create two columns for input and output
col1, col2 = st.columns([1, 2])

# Input column (left panel)
with col1:
    st.subheader("Input Parameters")
    
    exchange = st.selectbox("Exchange", ["OKX"], index=0)
    
    # Default pairs for OKX
    available_pairs = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]
    symbol = st.selectbox("Trading Pair", available_pairs, index=0)
    
    order_type = st.selectbox("Order Type", ["Market"], index=0)
    
    quantity = st.number_input("Quantity (USD equivalent)", 
                              min_value=10.0, 
                              max_value=1000.0, 
                              value=100.0, 
                              step=10.0,
                              help="Amount in USD to simulate trading")
    
    volatility = st.slider("Volatility (%)", 
                          min_value=0.1, 
                          max_value=5.0, 
                          value=1.0, 
                          step=0.1,
                          help="Higher values indicate more volatile market conditions")
    
    fee_tier = st.selectbox("Fee Tier", 
                           ["VIP 0 (0.080% Maker / 0.080% Taker)", 
                            "VIP 1 (0.070% Maker / 0.075% Taker)",
                            "VIP 2 (0.060% Maker / 0.070% Taker)"], 
                           index=0)
    
    # Extract fee values
    if "VIP 0" in fee_tier:
        maker_fee, taker_fee = 0.00080, 0.00080
    elif "VIP 1" in fee_tier:
        maker_fee, taker_fee = 0.00070, 0.00075
    else:
        maker_fee, taker_fee = 0.00060, 0.00070
    
    # Connection status and controls
    st.subheader("Connection")
    
    status_placeholder = st.empty()
    
    # Connection status indicator
    if st.session_state.connected:
        status_placeholder.success("Connected to WebSocket")
        if st.button("Disconnect"):
            if st.session_state.ws_client:
                st.session_state.ws_client.disconnect()
                st.session_state.connected = False
                st.session_state.ws_client = None
                st.session_state.processor = None
                st.rerun()
    else:
        status_placeholder.error("Not connected")
        if st.button("Connect"):
            try:
                # Create WebSocket client and processor
                websocket_url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{symbol}"
                
                # Initialize processor with models
                processor = OrderbookProcessor(
                    slippage_model=SlippageModel(),
                    fee_model=FeeModel(maker_fee, taker_fee),
                    impact_model=AlmgrenChrissModel(volatility/100),  # Convert percentage to decimal
                    maker_taker_model=MakerTakerModel()
                )
                
                # Initialize WebSocket client
                ws_client = WebSocketClient(
                    url=websocket_url,
                    on_message=lambda msg: processor.process_orderbook(json.loads(msg), quantity)
                )
                
                # Start connection in a separate thread
                ws_thread = threading.Thread(target=ws_client.connect, daemon=True)
                ws_thread.start()
                
                # Wait for connection to establish
                time.sleep(1)
                
                # Store in session state
                st.session_state.ws_client = ws_client
                st.session_state.processor = processor
                st.session_state.connected = True
                st.rerun()
                
            except Exception as e:
                logger.error(f"Connection error: {str(e)}")
                st.error(f"Connection error: {str(e)}")

# Output column (right panel)
with col2:
    st.subheader("Market Data & Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Transaction Cost Analysis", "Orderbook Visualization", "Performance Metrics", "Historical Data"])
    
    with tab1:
        # Transaction Cost Analysis
        output_container = st.container()
        
        with output_container:
            # Placeholder for metrics
            metrics_container = st.empty()
            
            # Create a placeholder for updating text
            update_info = st.empty()
            
            # If connected, continuously update the output parameters
            if st.session_state.connected and st.session_state.processor:
                # Display the metrics
                col1, col2 = metrics_container.columns(2)
                
                # Get the latest metrics from the processor
                metrics = st.session_state.processor.get_latest_metrics()
                
                if metrics:
                    # Left column metrics
                    with col1:
                        st.metric("Expected Slippage (%)", 
                                 format_number(metrics.get('slippage', 0) * 100, 4))
                        st.metric("Expected Fees (%)", 
                                 format_number(metrics.get('fees', 0) * 100, 4))
                        st.metric("Expected Market Impact (%)", 
                                 format_number(metrics.get('market_impact', 0) * 100, 4))
                    
                    # Right column metrics
                    with col2:
                        st.metric("Net Cost (%)", 
                                 format_number(metrics.get('net_cost', 0) * 100, 4))
                        st.metric("Maker/Taker Proportion", 
                                 f"{format_number(metrics.get('maker_proportion', 0) * 100, 2)}% / {format_number(metrics.get('taker_proportion', 0) * 100, 2)}%")
                        st.metric("Internal Latency (ms)", 
                                 format_number(metrics.get('processing_latency', 0) * 1000, 2))
                    
                    # Add last update time
                    update_info.info(f"Last updated: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                else:
                    metrics_container.info("Waiting for data...")
            else:
                metrics_container.info("Connect to see transaction cost analysis")
    
    with tab2:
        # Orderbook Visualization
        orderbook_container = st.container()
        
        with orderbook_container:
            chart_placeholder = st.empty()
            
            if st.session_state.connected and st.session_state.processor:
                # Get latest orderbook data
                orderbook = st.session_state.processor.get_latest_orderbook()
                
                if orderbook and len(orderbook.get('asks', [])) > 0 and len(orderbook.get('bids', [])) > 0:
                    # Prepare data for visualization
                    asks = np.array(orderbook['asks'], dtype=float)
                    bids = np.array(orderbook['bids'], dtype=float)
                    
                    # Create depth chart
                    fig = go.Figure()
                    
                    # Calculate cumulative volumes
                    asks_prices, asks_volumes = asks[:, 0], asks[:, 1]
                    bids_prices, bids_volumes = bids[:, 0], bids[:, 1]
                    
                    asks_cumulative = np.cumsum(asks_volumes)
                    bids_cumulative = np.cumsum(bids_volumes)
                    
                    # Add ask (sell) orders
                    fig.add_trace(go.Scatter(
                        x=asks_prices,
                        y=asks_cumulative,
                        mode='lines',
                        line=dict(width=2, color='red'),
                        name='Asks',
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.1)'
                    ))
                    
                    # Add bid (buy) orders
                    fig.add_trace(go.Scatter(
                        x=bids_prices,
                        y=bids_cumulative,
                        mode='lines',
                        line=dict(width=2, color='green'),
                        name='Bids',
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.1)'
                    ))
                    
                    # Calculate mid price
                    mid_price = (float(asks[0][0]) + float(bids[0][0])) / 2
                    
                    # Add spread line
                    fig.add_shape(
                        type="line",
                        x0=float(bids[0][0]), y0=0,
                        x1=float(asks[0][0]), y1=0,
                        line=dict(
                            color="yellow",
                            width=2,
                            dash="dot",
                        )
                    )
                    
                    # Add annotation for spread
                    spread = float(asks[0][0]) - float(bids[0][0])
                    spread_pct = (spread / mid_price) * 100
                    
                    fig.add_annotation(
                        x=mid_price,
                        y=0,
                        text=f"Spread: {format_number(spread, 2)} ({format_number(spread_pct, 4)}%)",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="yellow",
                        arrowsize=1,
                        arrowwidth=2,
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Orderbook Depth Chart - {orderbook['symbol']}",
                        xaxis_title="Price",
                        yaxis_title="Cumulative Volume",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0),
                        plot_bgcolor='rgba(17, 17, 17, 0.8)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        font=dict(color='white')
                    )
                    
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                else:
                    chart_placeholder.info("Waiting for orderbook data...")
            else:
                chart_placeholder.info("Connect to see orderbook visualization")

    with tab3:
        # Performance Metrics
        performance_container = st.container()
        
        with performance_container:
            perf_chart = st.empty()
            
            if st.session_state.connected and st.session_state.processor:
                # Get performance metrics from processor
                latency_history = st.session_state.processor.get_latency_history()
                
                if latency_history and len(latency_history) > 0:
                    # Convert to milliseconds for better readability
                    latency_ms = [l * 1000 for l in latency_history]
                    
                    # Create performance chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        y=latency_ms[-100:] if len(latency_ms) > 100 else latency_ms,
                        mode='lines',
                        name='Processing Latency',
                        line=dict(width=2, color='cyan')
                    ))
                    
                    # Calculate average, min and max
                    avg_latency = np.mean(latency_ms)
                    min_latency = np.min(latency_ms)
                    max_latency = np.max(latency_ms)
                    
                    # Add reference lines
                    fig.add_shape(type="line",
                        x0=0,
                        y0=avg_latency,
                        x1=len(latency_ms),
                        y1=avg_latency,
                        line=dict(
                            color="white",
                            width=1,
                            dash="dash",
                        )
                    )
                    
                    # Add annotation
                    fig.add_annotation(
                        x=0,
                        y=avg_latency,
                        text=f"Avg: {format_number(avg_latency, 2)} ms",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10, color="white")
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="Processing Latency Over Time",
                        xaxis_title="Updates",
                        yaxis_title="Latency (ms)",
                        showlegend=False,
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0),
                        plot_bgcolor='rgba(17, 17, 17, 0.8)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        font=dict(color='white')
                    )
                    
                    perf_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Latency (ms)", format_number(avg_latency, 2))
                    col2.metric("Min Latency (ms)", format_number(min_latency, 2))
                    col3.metric("Max Latency (ms)", format_number(max_latency, 2))
                    
                    # Messages per second calculation
                    if len(latency_ms) > 1:
                        total_time = st.session_state.processor.get_total_processing_time()
                        if total_time > 0:
                            msg_per_sec = len(latency_ms) / total_time
                            st.metric("Messages Processed per Second", format_number(msg_per_sec, 2))
                else:
                    perf_chart.info("Waiting for performance data...")
            else:
                perf_chart.info("Connect to see performance metrics")
                
    with tab4:
        # Historical Data Analysis
        render_history_tab()

# Auto-refresh mechanism for real-time updates (every 1 second)
if st.session_state.connected:
    time.sleep(1)
    st.rerun()   