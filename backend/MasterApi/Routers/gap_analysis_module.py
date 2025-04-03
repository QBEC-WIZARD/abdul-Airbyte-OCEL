## import all fasapi related libraries i want to make this a router forporcess-discoevery
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from tornado.websocket import WebSocketClosedError
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
import pandas as pd
import numpy as np
import io
import base64
import pandas as pd
import logging
from fastapi import Depends
from fastapi import Request

from backend.utils.helpers import extract_json_schema
from backend.utils.helpers import convert_timestamps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


gap_router = APIRouter(prefix="/v1/gap-analysis", tags=["Gap Analysis"])
# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')

# implement the root route
@gap_router.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the Gap Analysis Module"}

def generate_trading_volume_chart():
    """Creates a currency trading volume visualization and returns it as Base64 image."""
    import matplotlib.pyplot as plt

    # Mock Data for Trading Volume
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    eur_volume = [4000, 3000, 2000, 2780, 1890, 2390]
    usd_volume = [2400, 1398, 9800, 3908, 4800, 3800]
    gbp_volume = [2400, 2210, 2290, 2000, 2181, 2500]

    # Convert to NumPy for serialization safety
    dates = dates.strftime('%Y-%m-%d').tolist()  # Convert to string format
    eur_volume = np.array(eur_volume).tolist()
    usd_volume = np.array(usd_volume).tolist()
    gbp_volume = np.array(gbp_volume).tolist()

    # Create Plot
    plt.figure(figsize=(8, 5))
    plt.plot(dates, eur_volume, marker='o', label="EUR")
    plt.plot(dates, usd_volume, marker='s', label="USD")
    plt.plot(dates, gbp_volume, marker='^', label="GBP")
    plt.xlabel("Date")
    plt.ylabel("Trading Volume")
    plt.title("Currency Trading Volume Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Convert plot to Base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    return base64_image

@gap_router.get("/fx-analytics")
async def handle_graph_analytics():
    """Handle FX Trading Analytics (without visualization)"""

    # Mock Metrics Data
    trading_volume_chart = generate_trading_volume_chart()
    metrics = {
        "ml_model_accuracy": "94.5%",
        "anomalies_detected": 27,
        "pattern_confidence": "87.2%",
        "trading_volume": "$1.2B"
    }

    # Mock DataFrames (Converted to Dicts for JSON response)
    mock_volume = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=6, freq='M'),
        'EUR': [4000, 3000, 2000, 2780, 1890, 2390],
        'USD': [2400, 1398, 9800, 3908, 4800, 3800],
        'GBP': [2400, 2210, 2290, 2000, 2181, 2500]
    }).to_dict(orient="records")

    mock_patterns = pd.DataFrame({
        'time': ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
        'volume': [30, 45, 35, 50, 25, 40],
        'volatility': [65, 55, 85, 45, 60, 75]
    }).to_dict(orient="records")

    mock_anomalies = pd.DataFrame({
        'interval': list(range(1, 7)),
        'normal': [40, 30, 20, 27, 18, 23],
        'anomaly': [24, 13, 38, 15, 42, 19]
    }).to_dict(orient="records")

    # Return JSON Response
    return {
        "metrics": metrics,
        "trading_volume_chart": trading_volume_chart,
    }

