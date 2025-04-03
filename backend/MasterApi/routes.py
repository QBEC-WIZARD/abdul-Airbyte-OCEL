from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os
from computation.Process_discovery_module.Process_discovery_utils import create_directories
from computation.Process_discovery_module.Process_discovery_utils import save_uploaded_file
from computation.Process_discovery_module.Process_discovery_utils import analyze_risks
from computation.Process_discovery_module.Process_discovery_utils import visualize_risk_distribution
from computation.Process_discovery_module.Process_discovery_utils import show_loader
from computation.Process_discovery_module.Process_discovery_utils import hide_loader
from computation.Process_discovery_module.Process_discovery_utils import process_mining_analysis

from computation.Outlier_module.Outlier_Analysis_utils import get_object_interactions, get_object_metrics, get_object_lifecycle_graph
from computation.Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer

# import the OCELFailureMode class from the OCELFailure.py file
from computation.FMEA_module.OCELFailure import OCELFailureMode
# import the fmea_utils.py file
from computation.FMEA_module.fmea_utils import get_fmea_insights, display_rpn_distribution, display_fmea_analysis
# import the OCELDataManager class from the OCELDataManager.py file
from computation.FMEA_module.OCELDataManager import OCELDataManager
# import the OCELEnhancedFMEA class from the OCELEnhancedFMEA.py file
from computation.FMEA_module.OCELEnhancedFMEA import OCELEnhancedFMEA



from backend.utils.helpers import extract_json_schema
from backend.utils.helpers import convert_timestamps


from backend.models.pydantic_models import CSVResponse
from backend.models.pydantic_models import BPMNResponse
from backend.models.pydantic_models import ImageResponse
from backend.models.pydantic_models import  InteractionsResponse
from backend.models.pydantic_models import MetricModel
from backend.models.pydantic_models import LifecycleModel
from backend.models.pydantic_models import AIAnalysisResponse
from backend.models.pydantic_models import DataModel
from backend.models.pydantic_models import FMEAAnalysisResponse
from backend.models.pydantic_models import RPNDistributionPlot

#  import pm4py
import pm4py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
from fastapi import HTTPException
from typing import Dict
import pm4py
import logging
from fastapi import Depends
from pydantic import BaseModel,RootModel,Field
from typing import List
import tempfile
from utils import get_azure_openai_client
import os
import xml.dom.minidom
from fastapi import HTTPException, Response
import pm4py
import logging
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import os
import traceback
from collections import defaultdict
import json
import plotly.utils
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Set
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import tornado   # Required for Streamlit sharing
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
from typing import List
logger = logging.getLogger(__name__)


router = APIRouter()



# Pydantic model for CSV 


@router.post("/upload", response_model=CSVResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
    
    # Save the file to the api_response folder
    os.makedirs("api_response", exist_ok=True)
    save_path = os.path.join("api_response", "data_event_log.csv")
    with open(save_path, "wb") as f:
        f.write(content)
    
    return CSVResponse(message=f"Successfully uploaded {file.filename}", data=df.head().to_dict(orient="records"))









@router.get("/calculate_bpmn", response_model=BPMNResponse)
async def calculate():
    try:
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

@router.get("/fx_trade_process_tree_display",response_model=ImageResponse)
async def fx_trade_process_tree_display(process_tree_status: dict = Depends(check_fx_trade_process_tree_exists)):
    try:
        if not process_tree_status["exists"]:
            raise HTTPException(status_code=404, detail="Process tree image not found")

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

@router.get("/fx_trade_bpmn_display",response_model=ImageResponse)
async def fx_trade_bpmn_display(bpmn_status: dict = Depends(check_fx_trade_bpmn_exists)):
    try:
        if not bpmn_status["exists"]:
            raise HTTPException(status_code=404, detail="BPMN image not found")

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

@router.get("/fx_trade_petri_net_display",response_model=ImageResponse)
async def fx_trade_bpmn_display(petri_net_status: dict = Depends(check_fx_trade_petri_net_exists)):
    try:
        if not petri_net_status["exists"]:
            raise HTTPException(status_code=404, detail="BPMN image not found")

        return {
            "success": True,
            "status_code": 200,
            "message": "Successfully retrieved BPMN image",
            "image": petri_net_status["image"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

## THIS ROUTER ENDPOINT IS ONLY FOR THE STREAMLIT INPUT
@router.get("/download_csv")
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


## for the OUTLIER MODULE










## routing for ocpm ui
@router.get("/interactions",response_model=InteractionsResponse)
async def object_interactions():
    """API endpoint for object type interactions."""
    try:
        interactions_data = get_object_interactions()
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
        print('starting')
        print(extract_json_schema(interactions_list))
        
        return {"interactions": interactions_list}
    except Exception as e:
        print(f"Error in object_interactions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





@router.get("/metrics",response_model=MetricModel)
def object_metrics():
            """API endpoint for object type metrics."""
            try:
                metrics = get_object_metrics()
                return {"metrics": metrics}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))






@router.get("/lifecycle/{object_type}",response_model=LifecycleModel)
def object_lifecycle(object_type: str):
            """API endpoint for object lifecycle graph."""
            try:
                lifecycle_graph = get_object_lifecycle_graph(object_type)
                return {"lifecycle_graph": lifecycle_graph}  # Return as JSON
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


## OUTLIAER ANALYSIS MODULE AI INSIGHTS







@router.get('/run_ai_analysis',response_model=AIAnalysisResponse)
def run_ai_analysis():
    ocel_path = os.path.join("api_response", "process_data.json")
    if not os.path.exists(ocel_path):
        raise HTTPException(status_code=404, detail="process_data.json file not found")

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        stats = analyzer.stats

        # Default question for AI analysis
        default_question = "What are the main process patterns?"
        analysis_result = analyzer.analyze_with_ai(default_question)

        response = {
            "total_events": stats['general']['total_events'],
            "total_cases": stats['general']['total_cases'],
            "total_resources": stats['general']['total_resources'],
            "ai_analysis": analysis_result
        }

        return JSONResponse(response)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OCEL file not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during AI analysis: {str(e)}")





@router.get('/get_visualization_data',response_model=DataModel)
def get_visualization_data():
    """Retrieve data for process visualizations."""
    ocel_path = os.path.join("api_response", "process_data.json")
    if not os.path.exists(ocel_path):
        raise HTTPException(status_code=404, detail="process_data.json file not found")

    try:
        analyzer = IntegratedAPAAnalyzer()
        analyzer.load_ocel(ocel_path)

        # Generate visualizations
        figures = analyzer.create_visualizations()
        print(f"Type of figures: {type(figures)}")
        
        # Convert Plotly figures to JSON using plotly's built-in serialization

        
        # Convert Plotly figures to JSON-serializable format
        visualization_data = {
            "activity_distribution": json.loads(plotly.io.to_json(figures["activity_distribution"])),
            "resource_distribution": json.loads(plotly.io.to_json(figures["resource_distribution"]))
        }

        return JSONResponse(content=visualization_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return JSONResponse(content={"error": f"Error generating visualization data: {str(e)}"}, status_code=500)




# FX ANALYTICS MODULE

## DIGITAL TWIN CODE WITH GAP ANALYSIS

def generate_trading_volume_chart():
    """Creates a currency trading volume visualization and returns it as Base64 image."""

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

@router.get("/fx-analytics")
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



# FMEA ANALYSIS MODULE
# pages/5_FMEA_Analysis.py






def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')









@router.get("/fmea-analysis",response_model=FMEAAnalysisResponse)
def perform_fmea_analysis():
    try:
        # Verify OCEL model file exists
       # if not os.path.exists("api_response/output_ocel.json"): -->comeback if error is found
        if not os.path.exists("api_response/process_data.json"):
            raise HTTPException(status_code=400, detail="OCEL model file not found. Run Outlier Analysis first.")
        
        # Load process data
        with open("api_response/process_data.json", "r") as f:
            ocel_data = json.load(f)
        
        # Validate OCEL data structure
        if "ocel:events" not in ocel_data:
            logger.error("Invalid OCEL data structure")
            raise HTTPException(status_code=400, detail="Invalid OCEL data structure - missing events")
        
        # Initialize analyzer with validated data
        analyzer = OCELEnhancedFMEA(ocel_data)
        
        # Perform FMEA analysis
        fmea_results = analyzer.identify_failure_modes()
        # Save FMEA results to a JSON file
        fmea_results_path = os.path.join("api_response", "fmea_results.json")
        with open(fmea_results_path, "w") as f:
            json.dump(fmea_results, f, indent=4, default=convert_timestamps)
           # json.dump(fmea_results, f, indent=4)
        logger.info(f"Analysis complete. Found {len(fmea_results)} failure modes")
        
        # Get AI insights
        
        ai_insights = get_fmea_insights(fmea_results)
        # Save AI insights to a JSON file
        ai_insights_path = os.path.join("api_response", "ai_insights.json")
        with open(ai_insights_path, "w") as f:
            json.dump(ai_insights, f, indent=4)
        
        return JSONResponse(content={
            "failure_modes": len(fmea_results),
            "findings": ai_insights.get("findings", "No findings available"),
            "insights": ai_insights.get("insights", "No insights available"),
            "recommendations": ai_insights.get("recommendations", "No recommendations available"),
        })
    
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File error: {str(e)}")
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")




@router.get("/rpn_distribution",response_model=RPNDistributionPlot)
async def display_rpn_distribution():
    """
    Display comprehensive RPN distribution visualization with analysis breakdowns.
    This shows the spread of risk levels across different failure modes and helps
    identify risk clusters and patterns.
    """
    try:
        # Load FMEA results from a JSON file
        fmea_results_path = os.path.join("api_response", "fmea_results.json")
        if not os.path.exists(fmea_results_path):
            raise HTTPException(status_code=404, detail="FMEA results file not found")

        with open(fmea_results_path, "r") as f:
            fmea_results = json.load(f)

        # Create base RPN histogram
        df = pd.DataFrame(fmea_results)
        fig = go.Figure()

        # Add main RPN distribution histogram
        fig.add_trace(go.Histogram(
            x=df['rpn'],
            nbinsx=20,
            name='RPN Distribution',
            marker_color='blue',
            opacity=0.7
        ))

        # Add critical threshold line
        fig.add_vline(
            x=200,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold (RPN=200)",
            annotation_position="top right"
        )

        # Add risk zone annotations
        fig.add_vrect(
            x0=0, x1=100,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Low Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=100, x1=200,
            fillcolor="yellow", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Medium Risk",
            annotation_position="bottom"
        )
        fig.add_vrect(
            x0=200, x1=1000,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="High Risk",
            annotation_position="bottom"
        )

        # Update layout with detailed information
        fig.update_layout(
            title={
                'text': 'Risk Priority Number (RPN) Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='RPN Value',
            yaxis_title='Number of Failure Modes',
            showlegend=True,
            height=500,
            annotations=[
                dict(
                    x=50, y=1.05,
                    text="Low Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="green")
                ),
                dict(
                    x=150, y=1.05,
                    text="Medium Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="orange")
                ),
                dict(
                    x=250, y=1.05,
                    text="High Risk Zone",
                    showarrow=False,
                    xref='x', yref='paper',
                    font=dict(color="red")
                )
            ]
        )

        # Calculate distribution statistics
        stats = {
            "average_rpn": df['rpn'].mean(),
            "median_rpn": df['rpn'].median(),
            "std_deviation": df['rpn'].std(),
            "90th_percentile": df['rpn'].quantile(0.9)
        }

        # Risk zone analysis
        risk_zones = {
            'Low Risk (RPN ≤ 100)': len(df[df['rpn'] <= 100]),
            'Medium Risk (100 < RPN ≤ 200)': len(df[(df['rpn'] > 100) & (df['rpn'] <= 200)]),
            'High Risk (RPN > 200)': len(df[df['rpn'] > 200])
        }

        # Create risk zone bar chart
        risk_fig = go.Figure(data=[
            go.Bar(
                x=list(risk_zones.keys()),
                y=list(risk_zones.values()),
                marker_color=['green', 'yellow', 'red']
            )
        ])
        risk_fig.update_layout(
            title="Distribution by Risk Zone",
            xaxis_title="Risk Zone",
            yaxis_title="Number of Failure Modes",
            height=400
        )

        return JSONResponse(content={
            "rpn_distribution_plot": json.loads(fig.to_json()),
            "risk_zone_plot": json.loads(risk_fig.to_json()),
            "statistics": stats,
            "risk_zones": risk_zones
        })

    except Exception as e:
        logger.error(f"Error in display_rpn_distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error displaying RPN distribution: {str(e)}")

