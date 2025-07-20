Trade Simulator WebApp
The Trade Simulator WebApp is a real-time platform built in Python to simulate and analyze the performance of algorithmic trades using live Level 2 orderbook data from crypto exchanges like Binance or OKX. It connects via WebSocket, streams market depth in real time, and processes it to simulate trade executions under real market conditions.

The system computes key trading metrics such as slippage, fees, market impact, and applies advanced models like the Almgren-Chriss framework to optimize trade execution. Users can configure trade sizes and analyze results across multiple sessions.

The backend is structured with a modular, multi-threaded architecture in Python, integrating async WebSocket clients, custom regression models, and data processing pipelines. Core modules like OrderbookProcessor, WebSocketClient, and DatabaseManager ensure efficient data flow and storage. The app uses PostgreSQL for persisting orderbook snapshots, simulation logs, and performance metrics.

Additional utilities include VWAP calculation, latency analysis, and volume imbalance detection, offering deeper insights into trade efficiency and market behavior.

The project is packaged with Docker and managed via Poetry or pip, making it portable, reproducible, and easy to deploy in production environments or on the cloud.

So it was a task given to me by a firm name Qoquant when i had applied to them for internship where i had to create using websockets.It was first time when i used websockets in my projects.

