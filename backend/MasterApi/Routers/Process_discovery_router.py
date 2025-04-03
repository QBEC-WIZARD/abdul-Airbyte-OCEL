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

from computation.Process_discovery_module.Process_discovery_utils import create_directories
from computation.Process_discovery_module.Process_discovery_utils import save_uploaded_file
from computation.Process_discovery_module.Process_discovery_utils import analyze_risks
from computation.Process_discovery_module.Process_discovery_utils import visualize_risk_distribution
from computation.Process_discovery_module.Process_discovery_utils import show_loader
from computation.Process_discovery_module.Process_discovery_utils import hide_loader
from computation.Process_discovery_module.Process_discovery_utils import process_mining_analysis

#  import pm4py
import pm4py
import tempfile
import os
import xml.dom.minidom
from fastapi import HTTPException, Response
import pm4py
from fastapi import APIRouter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
from fastapi import HTTPException
import pm4py

import logging
from fastapi import Depends
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from backend.models.pydantic_models import CSVResponse
from backend.models.pydantic_models import BPMNResponse
from backend.models.pydantic_models import ImageResponse


from backend.utils.helpers import extract_json_schema
from backend.utils.helpers import convert_timestamps

import time
from .central_log import log_time

pd_router = APIRouter(prefix="/v1/process-discovery", tags=["Process Discovery"])

# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')

# implement the root route
@pd_router.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the Process Discovery Module"}

@pd_router.post("/upload-csv", response_model=CSVResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    content = await file.read()
    try:
        start=log_time("upload_csv", "START")
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
    
    # Save the file to the api_response folder
    os.makedirs("api_response", exist_ok=True)
    save_path = os.path.join("api_response", "data_event_log.csv")
    with open(save_path, "wb") as f:
        f.write(content)
    
    log_time("upload_csv", "END",start)
    return CSVResponse(message=f"Successfully uploaded {file.filename}", data=df.head().to_dict(orient="records"))





@pd_router.get("/calculate_bpmn", response_model=BPMNResponse)
async def calculate():
    try:
        start=log_time("calculate", "START")
        file_path = os.path.join("api_response", "data_event_log.csv")
        
        try:
            bpmn_graph, event_log = process_mining_analysis(file_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Data file not found")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bpmn")
        temp_file.close()
        
        pm4py.write_bpmn(bpmn_graph, temp_file.name)
        
        with open(temp_file.name, 'r') as f:
            bpmn_xml_raw = f.read()
        
        dom = xml.dom.minidom.parseString(bpmn_xml_raw)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        os.unlink(temp_file.name)
        log_time("calculate", "END",start)
        return BPMNResponse(
            success=True,
            status_code=200,
            message="Successfully generated BPMN XML",
            bpmn_xml=pretty_xml
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def check_fx_trade_process_tree_exists():
    file_path = os.path.join("api_response", "fx_trade_process_tree.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            image_content = f.read()
        base64_image = base64.b64encode(image_content).decode("utf-8")
        return {"exists": True, "image": base64_image}
    else:
        return {"exists": False, "image": None}

@pd_router.get("/fx_trade_process_tree_display",response_model=ImageResponse)
async def fx_trade_process_tree_display(process_tree_status: dict = Depends(check_fx_trade_process_tree_exists)):
    try:
        start=log_time("fx_trade_process_tree_display", "START")
        if not process_tree_status["exists"]:
            raise HTTPException(status_code=404, detail="Process tree image not found")

        log_time("fx_trade_process_tree_display", "END",start)
        return {
            "success": True,
            "status_code": 200,
            "message": "Successfully retrieved process tree image",
            "image": process_tree_status["image"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def check_fx_trade_bpmn_exists():
    file_path = os.path.join("api_response", "fx_trade_bpmn.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            image_content = f.read()
        base64_image = base64.b64encode(image_content).decode("utf-8")
        return {"exists": True, "image": base64_image}
    else:
        return {"exists": False, "image": None}

@pd_router.get("/fx_trade_bpmn_display",response_model=ImageResponse)
async def fx_trade_bpmn_display(bpmn_status: dict = Depends(check_fx_trade_bpmn_exists)):
    try:
        start=log_time("fx_trade_bpmn_display", "START")
        if not bpmn_status["exists"]:
            raise HTTPException(status_code=404, detail="BPMN image not found")
        log_time("fx_trade_bpmn_display", "END",start)
        return {
            "success": True,
            "status_code": 200,
            "message": "Successfully retrieved BPMN image",
            "image": bpmn_status["image"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def check_fx_trade_petri_net_exists():
    file_path = os.path.join("api_response", "fx_trade_petri_net.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            image_content = f.read()
        base64_image = base64.b64encode(image_content).decode("utf-8")
        return {"exists": True, "image": base64_image}
    else:
        return {"exists": False, "image": None}

@pd_router.get("/fx_trade_petri_net_display",response_model=ImageResponse)
async def fx_trade_petri_net_display(petri_net_status: dict = Depends(check_fx_trade_petri_net_exists)):
    try:
        start=log_time("fx_trade_petri_net_display", "START")
        if not petri_net_status["exists"]:
            raise HTTPException(status_code=404, detail="BPMN image not found")
        log_time("fx_trade_petri_net_display", "END",start)
        return {
            "success": True,
            "status_code": 200,
            "message": "Successfully retrieved BPMN image",
            "image": petri_net_status["image"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

## THIS ROUTER ENDPOINT IS ONLY FOR THE STREAMLIT INPUT
@pd_router.get("/download_csv")
async def download_csv():
    file_path = os.path.join("data","fx_trade_log_small.csv")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data.csv"}
    )
