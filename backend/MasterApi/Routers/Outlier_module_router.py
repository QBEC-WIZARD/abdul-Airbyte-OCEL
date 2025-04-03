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


#  import pm4py
import pm4py
import numpy as np
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
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import json
import plotly
from fastapi import Depends
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph

import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Dict, List, Any
import traceback

from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph


from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_failure_patterns
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_time_analysis
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_resource_analysis
from computation.Outlier_module.Outlier_Analysis_utils import initialize_unfair_ocel_analyzer_with_case_analysis_patterns

from computation.Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer
from computation.Outlier_module.Outlier_Analysis_utils import initialize_IntegratedAPAAnalyzer

from backend.models.pydantic_models import  InteractionsResponse
from backend.models.pydantic_models import MetricModel
from backend.models.pydantic_models import LifecycleModel
from backend.models.pydantic_models import AIAnalysisResponse
from backend.models.pydantic_models import DataModel
from backend.models.pydantic_models import ResourceAnalysis
from backend.models.pydantic_models import FailureLogic
from backend.models.pydantic_models import CaseAnalysisDocument
from backend.models.pydantic_models import TimeLogic


from backend.utils.helpers import extract_json_schema
from backend.utils.helpers import convert_timestamps
from .central_log import log_time

out_router = APIRouter(prefix="/v1/outlier-analysis", tags=["Outlier Analysis"])


# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')

# implement the root route
@out_router.get("/")
async def read_root(request: Request):
    return {"message": "Welcome to the Outlier Analysis Module"}

from computation.Outlier_module.Outlier_Analysis_utils import perform_analysis
from fastapi import Depends

cache_OCPM={}
@out_router.get("/initialize-OCPMAnalyzer")
async def OCPMAnalyzer_initialize():
    global cache_OCPM
    try:
        start=log_time("OCPMAnalyzer_initialize","START")
        cache_OCPM=perform_analysis()
        log_time("OCPMAnalyzer_initialize","END",start)
        return JSONResponse(content={"OCPM object created":"yes"},status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

def check_intialization_OCPM_object():
    global cache_OCPM
    return {"exists": bool(cache_OCPM)}

## routing for ocpm ui
@out_router.get("/object-iteractions",response_model=InteractionsResponse)
async def object_interactions(OCPM_status :dict=Depends(check_intialization_OCPM_object)):
    """API endpoint for object type interactions."""
    if not OCPM_status['exists']:
        raise HTTPException(status_code=404, detail="OCPM Object is not initialized ")
    try:
        start=log_time("object_interactions","START")

        interactions_data = get_object_interactions(cache_OCPM)
    
        print(f"Type of interactions: {type(interactions_data)}")
        print(f"Content of interactions: {interactions_data}")
        
        # Convert to a list format which is always JSON serializable
        interactions_list = []
        
        # Handle the case where interactions_data is already wrapped in a dict
        
        data_to_process = interactions_data["interactions"]
            
        # Process the data into a list of objects
        if isinstance(data_to_process, dict):
            for key, value in data_to_process.items():
                # Convert any type of key to a serializable format
                if hasattr(key, '__iter__') and not isinstance(key, str):
                    # Create a dictionary with elements as separate fields
                    interaction = {"elements": list(key), "count": value}
                else:
                    interaction = {"key": str(key), "count": value}
                interactions_list.append(interaction)

        # print(f"Interactions list: {interactions_list}")
      #  print('starting')
      #  print(extract_json_schema(interactions_list))
        log_time("object_interactions","END",start)
        return {"interactions": interactions_list}
    except Exception as e:
        print(f"Error in object_interactions: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@out_router.get("/object-metrics",response_model=MetricModel)
def object_metrics(OCPM_status :dict=Depends(check_intialization_OCPM_object)):
    """API endpoint for object type metrics."""
    if not OCPM_status['exists']:
        raise HTTPException(status_code=404, detail="OCPM Object is not initialized ")
    try:
        start=log_time("object_metrics","START")
        try:
            metrics = get_object_metrics(cache_OCPM)
        except Exception as e:
            raise HTTPException(status_code=500,detail="OCPM Object is not initialized")
        log_time("object_metrics","END",start)
        return {"metrics": metrics}  # Return as JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@out_router.get("/object-lifecycle/{object_type}",response_model=LifecycleModel)
async def object_lifecycle(object_type: str,OCPM_status :dict=Depends(check_intialization_OCPM_object)):
    """API endpoint for object lifecycle graph."""
    if not OCPM_status['exists']:
        raise HTTPException(status_code=404, detail="OCPM Object is not initialized ")
    try:
        start=log_time("object_lifecycle","START")
        try:
            lifecycle_graph = get_object_lifecycle_graph(cache_OCPM,object_type)
        except Exception as e:
            raise HTTPException(status_code=500,detail="OCPM Object is not initialized")
        log_time("object_lifecycle","END",start)
        return {"lifecycle_graph": lifecycle_graph}  # Return as JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


## OUTLIER ANALYSIS MODULE AI INSIGHTS



IntegratedAPAAnalyzer_object=None

def check_intialization_IntegratedAPAAnalyzer_object():
    global IntegratedAPAAnalyzer_object
    if not IntegratedAPAAnalyzer_object:
        return {"exists":False}
    else:
        return {"exists":True}

@out_router.get('/initialize-IntegratedAPAAnalyzer')
async def initializeIntegratedAPAAnalyzer():
    global IntegratedAPAAnalyzer_object
    try:
        start=log_time("initializeIntegratedAPAAnalyzer","START")
        IntegratedAPAAnalyzer_object=initialize_IntegratedAPAAnalyzer()
        log_time("initializeIntegratedAPAAnalyzer","END",start)
        return JSONResponse(content={"The IntegratedAPAAnalyzer_object initialized":"yes"},status_code=200)
    except Exception as e:
        raise HTTPException(status_code=404, detail="IntegratedAPAAnalyzer_object_analyzer is not initialized")


@out_router.get('/run_ai_analysis',response_model=AIAnalysisResponse)
def run_ai_analysis(IntegratedAPAAnalyzer_status:dict=Depends(check_intialization_IntegratedAPAAnalyzer_object)):
    if not IntegratedAPAAnalyzer_status['exists']:
        raise HTTPException(status_code=404, detail="IntegratedAPAAnalyzer_object is not initialized ")
    global IntegratedAPAAnalyzer_object_dict

    try:
        start=log_time("run_ai_analysis", "START")
        # analyzer = IntegratedAPAAnalyzer()
        # analyzer.load_ocel(ocel_path)

        stats = IntegratedAPAAnalyzer_object.stats

        # Default question for AI analysis
        default_question = "What are the main process patterns?"
        analysis_result = IntegratedAPAAnalyzer_object.analyze_with_ai(default_question)

        response = {
            "total_events": stats['general']['total_events'],
            "total_cases": stats['general']['total_cases'],
            "total_resources": stats['general']['total_resources'],
            "ai_analysis": analysis_result
        }
        log_time("run_ai_analysis", "END",start)
        return JSONResponse(response)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OCEL file not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during AI analysis: {str(e)}")




@out_router.get('/get_visualization_data',response_model=DataModel)
def get_visualization_data(IntegratedAPAAnalyzer_status:dict=Depends(check_intialization_IntegratedAPAAnalyzer_object)):
    if not IntegratedAPAAnalyzer_status['exists']:
        raise HTTPException(status_code=404, detail="IntegratedAPAAnalyzer_object is not initialized ")
    """Retrieve data for process visualizations."""
    ocel_path = os.path.join("api_response", "process_data.json")
    if not os.path.exists(ocel_path):
        raise HTTPException(status_code=404, detail="process_data.json file not found")

    try:
        start=log_time("get_visualization_data", "START")

        # Generate visualizations
        figures = IntegratedAPAAnalyzer_object.create_visualizations()
        print(f"Type of figures: {type(figures)}")
        
        # Convert Plotly figures to JSON using plotly's built-in serialization

        
        # Convert Plotly figures to JSON-serializable format
        visualization_data = {
            "activity_distribution": json.loads(plotly.io.to_json(figures["activity_distribution"])),
            "resource_distribution": json.loads(plotly.io.to_json(figures["resource_distribution"]))
        }
        log_time("get_visualization_data", "END",start)
        return JSONResponse(content=visualization_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return JSONResponse(content={"error": f"Error generating visualization data: {str(e)}"}, status_code=500)


#craete an end point router for _display_failure_patterns_markdown





from computation.Outlier_module.Outlier_Analysis_utils import convert_numpy_types
from computation.Outlier_module.Outlier_Analysis_utils import intitialize_UnfairOCELAnalyzer


UnfairOCELAnalyzer_object=None

def check_intialization_UnfairOCELAnalyzer_object():
    global UnfairOCELAnalyzer_object
    if UnfairOCELAnalyzer_object is None:
        return {"exists":False}
    else:
        return {"exists":True}


@out_router.get('/initialize-UnfairOCELAnalyzer')
async def initializeUnfairOCELAnalyzer():
    global UnfairOCELAnalyzer_object
    try:
        UnfairOCELAnalyzer_object=intitialize_UnfairOCELAnalyzer()
        print(type(UnfairOCELAnalyzer_object))
        return JSONResponse(content={"initializeUnfairOCELAnalyzer is initialized":"yes"})
    except Exception as e:
        return JSONResponse(content={f'The object could not be initialized due the error {str(e)}'})
        


@out_router.get('/display_failure_patterns',response_model=FailureLogic)
async def display_failure_patterns(UnfairOCELAnalyzer_object_status:dict=Depends(check_intialization_UnfairOCELAnalyzer_object)):
    """Display failure patterns."""
    if not UnfairOCELAnalyzer_object_status['exists']:
        raise HTTPException(status_code=404, detail="UnfairOCELAnalyzer_object is not initialized ")
    global UnfairOCELAnalyzer_object
    try:
        start=log_time("display_failure_patterns", "START")
        markdown_logic = initialize_unfair_ocel_analyzer_with_failure_patterns(UnfairOCELAnalyzer_object)
        log_time("display_failure_patterns", "END",start)
        return (markdown_logic)
      #  return {"markdown": markdown_logic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#create an endpoint for time_analysis of the failure patterns



@out_router.get('/resource_analysis',response_model=ResourceAnalysis)
async def resource_analysis(UnfairOCELAnalyzer_object_status:dict=Depends(check_intialization_UnfairOCELAnalyzer_object)):
    """Resource analysis of failure patterns."""
    if not UnfairOCELAnalyzer_object_status['exists']:
        raise HTTPException(status_code=404, detail="UnfairOCELAnalyzer_object is not initialized ")
    try:
        start=log_time("resource_analysis", "START")
        resource_analysis_data = initialize_unfair_ocel_analyzer_with_resource_analysis(UnfairOCELAnalyzer_object)
        log_time("resource_analysis", "END",start)
        return resource_analysis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@out_router.get('/time_analysis',response_model=TimeLogic)
async def time_analysis(UnfairOCELAnalyzer_object_status:dict=Depends(check_intialization_UnfairOCELAnalyzer_object)):
    """Time analysis of failure patterns."""
    if not UnfairOCELAnalyzer_object_status['exists']:
        raise HTTPException(status_code=404, detail="UnfairOCELAnalyzer_object is not initialized ")
    try:
        start=log_time("time_analysis", "START")
        time_analysis_data = initialize_unfair_ocel_analyzer_with_time_analysis(UnfairOCELAnalyzer_object)
        log_time("time_analysis", "END",start)
        return (time_analysis_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






@out_router.get('/case_analysis_patterns',response_model=CaseAnalysisDocument)
async def case_analysis_patterns(UnfairOCELAnalyzer_object_status:dict=Depends(check_intialization_UnfairOCELAnalyzer_object)):
    """Case analysis of failure patterns."""
    if not UnfairOCELAnalyzer_object_status['exists']:
        raise HTTPException(status_code=404, detail="UnfairOCELAnalyzer_object is not initialized ")
    try:
        start=log_time("case_analysis_patterns", "START")
        case_analysis_data = initialize_unfair_ocel_analyzer_with_case_analysis_patterns(UnfairOCELAnalyzer_object)
        log_time("case_analysis_patterns", "END",start)
        return case_analysis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

