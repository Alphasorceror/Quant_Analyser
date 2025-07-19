import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging

from data_manager import DataManager
from utils import format_number

# Configure logging
logger = logging.getLogger(__name__)

def load_historical_simulations(symbol=None, limit=100):
    """Load historical simulation results from database"""
    data_manager = DataManager()
    simulations = data_manager.get_historical_simulations(symbol, limit)
    
    # Convert to pandas DataFrame
    if simulations:
        data = []
        for sim in simulations:
            data.append({
                'id': sim.id,
                'timestamp': sim.timestamp,
                'exchange': sim.exchange,
                'symbol': sim.symbol,
                'quantity': sim.quantity,
                'slippage': sim.slippage * 100,  # Convert to percentage
                'fees': sim.fees * 100,  # Convert to percentage
                'market_impact': sim.market_impact * 100,  # Convert to percentage
                'net_cost': sim.net_cost * 100,  # Convert to percentage
                'maker_proportion': sim.maker_proportion * 100,  # Convert to percentage
                'mid_price': sim.mid_price
            })
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

def load_performance_metrics(metric_name=None, limit=1000):
    """Load performance metrics from database"""
    data_manager = DataManager()
    metrics = data_manager.get_performance_metrics(metric_name, limit)
    
    # Convert to pandas DataFrame
    if metrics:
        data = []
        for metric in metrics:
            data.append({
                'id': metric.id,
                'timestamp': metric.timestamp,
                'metric_name': metric.metric_name,
                'metric_value': metric.metric_value
            })
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

def load_recent_orderbooks(symbol=None, limit=10):
    """Load recent orderbook snapshots from database"""
    data_manager = DataManager()
    orderbooks = data_manager.get_recent_orderbooks(symbol, limit)
    
    # Convert to list of dictionaries
    if orderbooks:
        data = []
        for ob in orderbooks:
            bid_levels = []
            ask_levels = []
            
            try:
                if ob.bid_levels_json and isinstance(ob.bid_levels_json, str):
                    bid_levels = json.loads(ob.bid_levels_json)
                if ob.ask_levels_json and isinstance(ob.ask_levels_json, str):
                    ask_levels = json.loads(ob.ask_levels_json)
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
            
            data.append({
                'id': ob.id,
                'timestamp': ob.timestamp,
                'exchange': ob.exchange,
                'symbol': ob.symbol,
                'mid_price': ob.mid_price,
                'best_bid': ob.best_bid,
                'best_ask': ob.best_ask,
                'spread': ob.spread,
                'bid_depth': ob.bid_depth,
                'ask_depth': ob.ask_depth,
                'bid_levels': bid_levels,
                'ask_levels': ask_levels
            })
        return data
    else:
        return []

def render_history_tab():
    """Render the history tab in the UI"""
    st.subheader("Historical Data Analysis")
    
    # Create subtabs
    subtab1, subtab2, subtab3 = st.tabs(["Simulation History", "Performance Metrics", "Orderbook Snapshots"])
    
    with subtab1:
        st.subheader("Simulation Results History")
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            symbol_filter = st.selectbox(
                "Filter by Symbol", 
                ["All", "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"],
                index=0
            )
        
        with col2:
            limit = st.slider("Number of records", 10, 200, 50)
        
        # Get data
        symbol = None if symbol_filter == "All" else symbol_filter
        df = load_historical_simulations(symbol, limit)
        
        if not df.empty:
            # Show data table
            st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
            
            # Create time series chart
            st.subheader("Cost Components Over Time")
            
            # Prepare data for line chart
            df_plot = df.copy()
            df_plot['date'] = pd.to_datetime(df_plot['timestamp'])
            df_plot = df_plot.sort_values(by='date')
            
            # Create figure
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['slippage'], 
                name='Slippage',
                line=dict(width=2, color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['fees'], 
                name='Fees',
                line=dict(width=2, color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['market_impact'], 
                name='Market Impact',
                line=dict(width=2, color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['net_cost'], 
                name='Net Cost',
                line=dict(width=3, color='purple')
            ))
            
            # Update layout
            fig.update_layout(
                title="Trading Costs Over Time",
                xaxis_title="Date",
                yaxis_title="Cost (%)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a price chart
            st.subheader("Price History")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['mid_price'], 
                name='Mid Price',
                line=dict(width=2, color='cyan')
            ))
            
            fig2.update_layout(
                title="Mid Price Over Time",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No historical simulation data available yet. Connect to the data feed and run some simulations.")
    
    with subtab2:
        st.subheader("Performance Metrics History")
        
        # Filter controls
        metric_filter = st.selectbox(
            "Select Metric", 
            ["processing_latency"],
            index=0
        )
        
        # Get data
        df = load_performance_metrics(metric_filter)
        
        if not df.empty:
            # Convert to milliseconds for better readability if latency
            if metric_filter == "processing_latency":
                df['metric_value'] = df['metric_value'] * 1000  # Convert to ms
                y_axis_title = "Latency (ms)"
            else:
                y_axis_title = "Value"
                
            # Prepare data for line chart
            df_plot = df.copy()
            df_plot['date'] = pd.to_datetime(df_plot['timestamp'])
            df_plot = df_plot.sort_values(by='date')
            
            # Create latency histogram
            fig1 = px.histogram(
                df_plot, 
                x='metric_value',
                nbins=30,
                title="Distribution of Processing Latency"
            )
            
            fig1.update_layout(
                xaxis_title=y_axis_title,
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create time series chart
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['metric_value'], 
                mode='lines',
                name=metric_filter,
                line=dict(width=1.5, color='cyan')
            ))
            
            # Calculate moving average
            window_size = min(20, len(df_plot))
            if window_size > 1:
                df_plot['ma'] = df_plot['metric_value'].rolling(window=window_size).mean()
                
                fig2.add_trace(go.Scatter(
                    x=df_plot['date'], 
                    y=df_plot['ma'], 
                    mode='lines',
                    name=f'{window_size}-point Moving Average',
                    line=dict(width=2, color='yellow')
                ))
            
            # Update layout
            fig2.update_layout(
                title=f"{metric_filter.replace('_', ' ').title()} Over Time",
                xaxis_title="Date",
                yaxis_title=y_axis_title,
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average", format_number(df_plot['metric_value'].mean(), 2))
            col2.metric("Minimum", format_number(df_plot['metric_value'].min(), 2))
            col3.metric("Maximum", format_number(df_plot['metric_value'].max(), 2))
            col4.metric("95th Percentile", format_number(df_plot['metric_value'].quantile(0.95), 2))
            
        else:
            st.info("No performance metrics data available yet. Connect to the data feed and run some simulations.")
    
    with subtab3:
        st.subheader("Orderbook Snapshots")
        
        # Filter controls
        symbol_filter = st.selectbox(
            "Select Symbol", 
            ["All", "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"],
            index=0,
            key="ob_symbol_filter"
        )
        
        # Get data
        symbol = None if symbol_filter == "All" else symbol_filter
        orderbooks = load_recent_orderbooks(symbol, 10)
        
        if orderbooks:
            # Create selection widget
            timestamps = [ob['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for ob in orderbooks]
            selected_ts = st.selectbox("Select Snapshot", timestamps)
            
            # Find selected orderbook
            selected_ob = next((ob for ob in orderbooks if ob['timestamp'].strftime('%Y-%m-%d %H:%M:%S') == selected_ts), None)
            
            if selected_ob:
                # Display orderbook info
                col1, col2, col3 = st.columns(3)
                col1.metric("Mid Price", format_number(selected_ob['mid_price'], 2))
                col2.metric("Spread", format_number(selected_ob['spread'], 4))
                col3.metric("Spread %", format_number((selected_ob['spread'] / selected_ob['mid_price']) * 100, 4) + "%")
                
                # Create depth chart if data is available
                if selected_ob['ask_levels'] and selected_ob['bid_levels']:
                    # Convert to numpy arrays
                    asks = np.array(selected_ob['ask_levels'], dtype=float)
                    bids = np.array(selected_ob['bid_levels'], dtype=float)
                    
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
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Orderbook Depth Chart - {selected_ob['symbol']} at {selected_ts}",
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display buy/sell imbalance
                    bid_volume = selected_ob['bid_depth']
                    ask_volume = selected_ob['ask_depth']
                    total_volume = bid_volume + ask_volume
                    
                    if total_volume > 0:
                        bid_percentage = (bid_volume / total_volume) * 100
                        ask_percentage = (ask_volume / total_volume) * 100
                        
                        st.subheader("Order Volume Distribution")
                        
                        # Create a horizontal stacked bar chart
                        fig2 = go.Figure()
                        
                        fig2.add_trace(go.Bar(
                            y=["Volume"],
                            x=[bid_percentage],
                            name='Bid Volume',
                            orientation='h',
                            marker=dict(color='green')
                        ))
                        
                        fig2.add_trace(go.Bar(
                            y=["Volume"],
                            x=[ask_percentage],
                            name='Ask Volume',
                            orientation='h',
                            marker=dict(color='red')
                        ))
                        
                        fig2.update_layout(
                            barmode='stack',
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor='rgba(17, 17, 17, 0.8)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            font=dict(color='white'),
                            showlegend=False,
                            xaxis=dict(
                                title="Percentage (%)",
                                range=[0, 100]
                            )
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Bid Volume", format_number(bid_volume, 2))
                        col2.metric("Ask Volume", format_number(ask_volume, 2))
                        
                        # Calculate imbalance (-1 to 1, negative means more asks, positive means more bids)
                        imbalance = (bid_volume - ask_volume) / total_volume
                        col3.metric("Imbalance", format_number(imbalance * 100, 2) + "%")
                
            else:
                st.warning("Selected orderbook not found")
        else:
            st.info("No orderbook snapshot data available yet. Connect to the data feed and run some simulations.")