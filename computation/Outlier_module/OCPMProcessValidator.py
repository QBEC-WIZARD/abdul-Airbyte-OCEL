import traceback
from collections import defaultdict


import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import warnings
import logging

# initializa logger
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from backend.MasterApi.Routers.central_log import log_time

from utils import get_azure_openai_client

class OCPMProcessValidator:
    """Handles validation of process execution using OCPM model and thresholds"""

    def __init__(self):
        """Initialize validator with OCPM model and thresholds"""
        try:
            start = log_time("__init__ OCPMProcessValidator", "START")
            # Load OCPM model
            with open('api_response/output_ocel.json', 'r') as f:
                self.ocpm_model = json.load(f)

            # Load timing thresholds
            with open('api_response/output_ocel_threshold.json', 'r') as f:
                self.thresholds = json.load(f)

            # Validate both files exist and have required structure
            self._validate_loaded_data()
            log_time("__init__ OCPMProcessValidator", "END", start)
        except Exception as e:
            logger.error(f"Error initializing OCPMProcessValidator: {str(e)}")
            raise

    def _validate_loaded_data(self):
        """Validates that loaded data has required structure"""
        start = log_time("_validate_loaded_data", "START")
        # Validate OCPM model
        for obj_type, data in self.ocpm_model.items():
            required_keys = {'activities', 'attributes', 'relationships'}
            if not all(key in data for key in required_keys):
                raise ValueError(f"Invalid OCPM model structure for {obj_type}")

        # Validate thresholds
        for obj_type, data in self.thresholds.items():
            required_keys = {'total_duration_hours', 'default_gap_hours', 'activity_thresholds'}
            if not all(key in data for key in required_keys):
                raise ValueError(f"Invalid threshold structure for {obj_type}")
        log_time("_validate_loaded_data", "END", start)


    def get_expected_flow(self) -> Dict[str, List[str]]:
        """Convert OCPM model to expected flow"""
        start = log_time("get_expected_flow", "START")
        log_time("get_expected_flow", "END", start)
        return {
            obj_type: data['activities']
            for obj_type, data in self.ocpm_model.items()
        }

    def get_timing_thresholds(self) -> Dict[str, Dict]:
        """Convert OCPM thresholds to timing rules"""
        return {
            obj_type: {
                'total_duration': data['total_duration_hours'],
                'activity_gaps': {
                    activity: thresholds['max_gap_after_hours']
                    for activity, thresholds in data['activity_thresholds'].items()
                }
            }
            for obj_type, data in self.thresholds.items()
        }
