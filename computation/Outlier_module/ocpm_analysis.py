# ocpm_analysis.py
import traceback
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
# import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import json
import requests
import json
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import logging

logger = logging.getLogger(__name__)




from computation.Outlier_module.azure_processor import process_log_file
from utils import get_azure_openai_client

# Add this at the start of your script, before any other imports
os.environ["PATH"] += os.pathsep + r"C:\samadhi\technology\Graphviz\bin"

from backend.MasterApi.Routers.central_log import log_time

@dataclass
class ObjectType:
    """Represents an object type in the OCPM model"""
    name: str
    activities: List[str]
    attributes: List[str]
    relationships: List[str]


class OCPMAnalyzer:
    """Main class for Object-Centric Process Mining analysis"""

    def __init__(self, event_log: pd.DataFrame):
        start = log_time("__init__ OCPMAnalyzer", "START")
        # Log the original columns for debugging
        print(f"Original columns: {event_log.columns.tolist()}")

        self.event_log                  = self._preprocess_event_log(event_log)
        self.object_types               = self._initialize_object_types()
        self.object_relationships       = defaultdict(list)
        self.activity_object_mapping    = self._create_activity_mapping()
        self.ocpm_df                    = self.convert_to_ocpm_format()

        log_time("__init__ OCPMAnalyzer", "END", start)


    def convert_to_ocel(self) -> Dict:
        
        """Convert OCPM data to OCEL format."""
        start = log_time("convert_to_ocel", "START")

        # ocpm_df = self.convert_to_ocpm_format()
        ocpm_df = self.ocpm_df
        ocel_data = {
            "ocel:version": "1.0",
            "ocel:ordering": "timestamp",
            "ocel:attribute-names": [],
            "ocel:events": [],
            "ocel:objects": [],
            "ocel:object-types": list(self.object_types.keys()),
            "ocel:global-log": {
                "ocel:attribute-names": []
            }
        }

        # Convert events
        events_map = {}  # To track unique events
        for _, row in ocpm_df.iterrows():
            event_id = f"{row['case_id']}_{row['activity']}"
            if event_id not in events_map:
                event = {
                    "ocel:id": event_id,
                    "ocel:timestamp": row['timestamp'].isoformat(),
                    "ocel:activity": row['activity'],
                    "ocel:type": "event",
                    "ocel:attributes": {
                        "resource": row['resource'],
                        "case_id": row['case_id'],
                        "object_type": row['object_type']
                    },
                    "ocel:objects": [{
                        "id": row['object_id'],
                        "type": row['object_type']
                    }]
                }
                ocel_data["ocel:events"].append(event)
                events_map[event_id] = True

        # Convert objects
        objects_map = {}  # To track unique objects
        for obj_type, obj_info in self.object_types.items():
            for activity in obj_info.activities:
                obj_id = f"{obj_type}_{activity}"
                if obj_id not in objects_map:
                    obj = {
                        "ocel:id": obj_id,
                        "ocel:type": obj_type,
                        "ocel:attributes": {
                            "activity": activity,
                            "attributes": obj_info.attributes
                        }
                    }
                    ocel_data["ocel:objects"].append(obj)
                    objects_map[obj_id] = True

        log_time("convert_to_ocel", "END", start)

        return ocel_data



    def save_ocel(self) -> str:
        """Save OCPM data in OCEL format."""

        try:
            start = log_time("save_ocel", "START")
            # Generate OCEL data
            ocel_data = self.convert_to_ocel()

            # Debug output
            logger.debug("Generated OCEL data structure:")
            logger.debug("Top-level keys: %s", list(ocel_data.keys()))
            logger.debug("Number of events: %d", len(ocel_data.get('ocel:events', [])))

            # Define output path
            # output_path = Path("ocpm_output/process_data.json")
            api_response_path = Path("api_response/process_data.json")

            # Create output directory if needed
            # output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save OCEL file
            # with open(output_path, 'w', encoding='utf-8') as f:
            #     json.dump(ocel_data, f, indent=2, ensure_ascii=False)
            with open(api_response_path, 'w', encoding='utf-8') as f:
                self.ocel_data=ocel_data # for the ocel_data specifically
                json.dump(ocel_data, f, indent=2, ensure_ascii=False)


            # Verify the file
            with open(api_response_path, 'r', encoding='utf-8') as f:
                verification = json.load(f)
                if not verification.get('ocel:events'):
                    logger.error("Generated OCEL file has no events")
                    raise ValueError("Generated OCEL file has no events")
                else:
                    logger.info("Successfully saved %d events", len(verification['ocel:events']))

            
            log_time("save_ocel", "END", start)

            return str(api_response_path)

        except Exception as e:
            logger.error("Error saving OCEL file: %s", str(e))
            raise



    def _preprocess_event_log(self, df: pd.DataFrame) -> pd.DataFrame:
        start = log_time("_preprocess_event_log", "START")
        """Preprocess event log to ensure consistent format"""


        # Create a copy to avoid modifying the original
        df = df.copy()

        # Check if we need to handle semicolon-separated data that was read as a single column
        if len(df.columns) == 1 and ';' in df.columns[0]:
            # Split the single column into multiple columns
            column_names = df.columns[0].split(';')
            df = pd.DataFrame([row[0].split(';') for row in df.values], columns=column_names)

        # Define all possible column mappings
        column_mappings = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'Case_': 'case_id',
            'Activity': 'activity',
            'Timestamp': 'timestamp',
            'case': 'case_id',
            'event': 'activity',
            'start_timestamp': 'timestamp'
        }

        # Rename columns if they exist
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Check required columns
        required_columns = ['case_id', 'activity', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error("Available columns: %s", df.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info("Processed DataFrame Info:")
        logger.info(df.info())
        log_time("_preprocess_event_log", "END", start)
        return df
    def _preprocess_event_log_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess event log to ensure consistent format with improved performance"""
        start = log_time("_preprocess_event_log_optimized", "START")

        # Fast copy to avoid modifying original
        df = df.copy(deep=False)

        # Efficient column handling for semicolon-separated data
        if len(df.columns) == 1 and ';' in str(df.columns[0]):
            # Use more efficient splitting method
            df = pd.DataFrame([row[0].split(';') for row in df.values], 
                            columns=df.columns[0].split(';'))

        # Optimized column mapping with faster lookup
        COLUMN_MAPPINGS = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'Case_': 'case_id',
            'Activity': 'activity',
            'Timestamp': 'timestamp',
            'case': 'case_id',
            'event': 'activity',
            'start_timestamp': 'timestamp'
        }

        # Vectorized column renaming
        df.rename(columns={old: new for old, new in COLUMN_MAPPINGS.items() 
                        if old in df.columns}, 
                inplace=True)

        # Fast required columns check
        required_columns = ['case_id', 'activity', 'timestamp']
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error("Available columns: %s", list(df.columns))
            raise ValueError(f"Missing required columns: {list(missing)}")

        # Optimized timestamp conversion
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
        except ValueError:
            logger.error("Invalid timestamp format")
            raise

        # Optional: Minimal logging to reduce overhead
        logger.debug(f"Processed DataFrame: {len(df)} rows")

        log_time("_preprocess_event_log_optimized", "END", start)
        return df

        
    def generate_timing_thresholds(self) -> Dict:
        """
        Generates timing thresholds based on OCEL object model.
        This method should be called after process_log_file has generated output_ocel.json
        """
        try:
            start = log_time("generate_timing_thresholds", "START")
            # First load the existing OCEL model
            with open('api_response/output_ocel.json', 'r') as f:
                ocel_model = json.load(f)

            # Initialize OpenAI client
            client = get_azure_openai_client()

            # Create prompt for timing threshold generation
            prompt = f"""
            Based on this OCEL object model, generate appropriate timing thresholds for a financial trading system:

            {json.dumps(ocel_model, indent=2)}

            Generate timing thresholds following these rules:
            1. Each object type needs:
               - Total maximum duration for all activities (in hours)
               - Default maximum gap between consecutive activities (in hours)
               - Activity-specific thresholds for processing time and gaps

            2. Consider these factors:
               - Trading activities should have shorter durations (typically < 1 hour)
               - Risk and compliance activities can have longer durations
               - Market data activities should be near real-time
               - Settlement activities can span multiple hours

            Return only valid JSON following this exact structure:
            {{
              "ObjectType": {{
                "total_duration_hours": number,
                "default_gap_hours": number,
                "activity_thresholds": {{
                  "activity_name": {{
                    "max_duration_hours": number,
                    "max_gap_after_hours": number
                  }}
                }}
              }}
            }}
            """

            # Get threshold recommendations from OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in financial trading processes and regulatory compliance. Provide timing thresholds that reflect real-world trading operations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            # Parse and validate the response
            thresholds = json.loads(response.choices[0].message.content)

            # Validate the structure matches our requirements
            for obj_type, config in thresholds.items():
                required_keys = {'total_duration_hours', 'default_gap_hours', 'activity_thresholds'}
                if not all(key in config for key in required_keys):
                    raise ValueError(f"Missing required keys in threshold config for {obj_type}")

            # Save the thresholds
            os.makedirs('api_response', exist_ok=True)
            api_response="api_response/output_ocel_threshold.json"
            with open(api_response, 'w') as f:
                json.dump(thresholds, f, indent=2)



            # os.makedirs('ocpm_output', exist_ok=True)
            # with open('ocpm_output/output_ocel_threshold.json', 'w') as f:
            #     json.dump(thresholds, f, indent=2)

            # Send thresholds to API endpoint
            # api_url = "http://127.0.0.1:8000/output_ocel_threshold"  # Replace with your API endpoint URL
            # try:
            #     response = requests.post(api_url, json=thresholds)
            #     response.raise_for_status()
            #     print("Thresholds successfully sent to API endpoint")
            # except requests.exceptions.RequestException as e:
            #     print(f"Failed to send thresholds to API endpoint: {str(e)}")

            log_time("generate_timing_thresholds", "END", start)
            return thresholds

        except Exception as e:
            print(f"Error generating timing thresholds: {str(e)}")
            print(traceback.format_exc())
            raise


    def _initialize_object_types(self, use_azure: bool = True) -> Dict[str, ObjectType]:
        """Initialize object types, optionally using Azure OpenAI"""
        try:
            start = log_time("_initialize_object_types", "START")
            if use_azure:
                # Process directly with Azure
                azure_types = process_log_file(df=self.event_log,chunk_size=1000)

                # Generate timing thresholds based on the object model
                self.generate_timing_thresholds()
                log_time("_initialize_object_types", "END", start)
                return azure_types
            log_time("_initialize_object_types", "END", start)
            return self._get_default_object_types()
            
        except Exception as e:
            print(f"Error in object type initialization: {str(e)}")
            return self._get_default_object_types()

    def _get_default_object_types(self) -> Dict[str, ObjectType]:
        start = log_time("_get_default_object_types", "START")
        """Return default object types"""
        print("Using default object types")
        log_time("_get_default_object_types", "END", start)
        return {
            'Trade': ObjectType(
                name='Trade',
                activities=[
                    'Trade Initiated', 'Trade Executed', 'Trade Allocated',
                    'Trade Settled', 'Trade Canceled'
                ],
                attributes=['currency_pair', 'notional_amount'],
                relationships=['Market', 'Risk', 'Settlement']
            ),
            'Market': ObjectType(
                name='Market',
                activities=[
                    'Trade Executed', 'Quote Requested', 'Quote Provided'
                ],
                attributes=['currency_pair'],
                relationships=['Trade']
            ),
            'Risk': ObjectType(
                name='Risk',
                activities=[
                    'Trade Allocated', 'Risk Assessment'
                ],
                attributes=['risk_score'],
                relationships=['Trade', 'Settlement']
            ),
            'Settlement': ObjectType(
                name='Settlement',
                activities=[
                    'Trade Settled', 'Position Reconciliation'
                ],
                attributes=['settlement_amount'],
                relationships=['Trade', 'Risk']
            )
        }


    def _create_activity_mapping(self) -> Dict[str, List[str]]:
        start = log_time("_create_activity_mapping", "START")
        """Create mapping of activities to object types"""
        mapping = defaultdict(list)
        for obj_type, obj_info in self.object_types.items():
            for activity in obj_info.activities:
                mapping[activity].append(obj_type)
        log_time("_create_activity_mapping", "END", start)
        return mapping

    def convert_to_ocpm_format(self) -> pd.DataFrame:
        """Convert regular event log to OCPM format"""  
        start = log_time("convert_to_ocpm_format", "START")
        ocpm_events = []

        default_object_type = next(iter(self.object_types.keys())) if self.object_types else None
        if not default_object_type:
            raise ValueError("No object types available for conversion")

        print(f"Using default object type: {default_object_type}")

        for _, row in self.event_log.iterrows():
            activity = row['activity']
            related_objects = self.activity_object_mapping.get(activity, [default_object_type])

            for obj_type in related_objects:
                event = {
                    'case_id': row['case_id'],
                    'activity': activity,
                    'timestamp': row['timestamp'],
                    'object_type': obj_type,
                    'object_id': f"{obj_type}_{row['case_id']}",
                    'resource': row.get('resource', 'Unknown'),
                }

                # Add object-specific attributes
                for attr in self.object_types[obj_type].attributes:
                    if attr in row:
                        event[f"{obj_type.lower()}_{attr}"] = row[attr]

                ocpm_events.append(event)

                # Record object relationships
                for related_obj in self.object_types[obj_type].relationships:
                    self.object_relationships[obj_type].append(related_obj)


        log_time("convert_to_ocpm_format", "END", start)

        return pd.DataFrame(ocpm_events)

    def analyze_object_interactions(self) -> Dict:
        start=log_time("analyze_object_interactions","START")
        """Analyze interactions between different object types"""
        interactions = defaultdict(int)

        for obj_type, related_objects in self.object_relationships.items():
            for related_obj in related_objects:
                key = tuple(sorted([obj_type, related_obj]))
                interactions[key] += 1
        log_time("analyze_object_interactions","END",start)
        
        return dict(interactions)

    def calculate_object_metrics(self) -> Dict:
        """Calculate comprehensive metrics for each object type"""
        start=log_time("calculate_object_metrics","START")
        metrics = {}
        # ocpm_df = self.convert_to_ocpm_format()
        ocpm_df = self.ocpm_df

        for obj_type in self.object_types.keys():
            obj_data = ocpm_df[ocpm_df['object_type'] == obj_type]

            unique_objects = len(obj_data['object_id'].unique())
            avg_activities = len(obj_data) / unique_objects if unique_objects > 0 else 1

            metrics[obj_type] = {
                'total_instances': len(obj_data['object_id'].unique()),
                'total_activities': len(obj_data['activity'].unique()),
                'avg_activities_per_instance': avg_activities,
                'top_activities': obj_data['activity'].value_counts().head(3).to_dict(),
                'interaction_count': len(self.object_relationships[obj_type])
            }
        log_time("calculate_object_metrics","END",start)
        return metrics

    def generate_object_lifecycle_graph(self, object_type: str) -> nx.DiGraph:
        start=log_time("generate_object_lifecycle_graph","START")
        """Generate lifecycle graph for specific object type"""
        # ocpm_df = self.convert_to_ocpm_format()
        ocpm_df = self.ocpm_df
        obj_data = ocpm_df[ocpm_df['object_type'] == object_type]

        G = nx.DiGraph()

        # Add nodes and edges based on activity sequence
        for _, group in obj_data.groupby('object_id'):
            activities = group['activity'].tolist()
            for i in range(len(activities) - 1):
                G.add_edge(activities[i], activities[i + 1])
        log_time("generate_object_lifecycle_graph","END",start)
        return G


class OCPMVisualizer:
    """Handles visualization of OCPM analysis results"""

    @staticmethod
    def create_object_interaction_heatmap(interactions: Dict[Tuple[str, str], int]) -> go.Figure:
        start = log_time("create_object_interaction_heatmap", "START")
        """Create heatmap of object interactions"""
        object_types = sorted(list(set([obj for pair in interactions.keys() for obj in pair])))
        matrix = np.zeros((len(object_types), len(object_types)))

        for (obj1, obj2), count in interactions.items():
            i, j = object_types.index(obj1), object_types.index(obj2)
            matrix[i][j] = count
            matrix[j][i] = count

        log_time("create_object_interaction_heatmap", "END", start)

        return go.Figure(data=go.Heatmap(
            z=matrix,
            x=object_types,
            y=object_types,
            colorscale='Viridis'
        ))

    @staticmethod
    def create_object_metrics_dashboard(metrics: Dict) -> List[go.Figure]:
        """Create dashboard visualizations for object metrics"""
        start = log_time("create_object_metrics_dashboard", "START")
        figures = []

        # Activity Distribution
        activity_counts = {obj: data['total_activities']
                           for obj, data in metrics.items()}

        figures.append(go.Figure(data=[
            go.Bar(x=list(activity_counts.keys()),
                   y=list(activity_counts.values()),
                   name='Activities per Object Type')
        ]))

        # Instance Distribution
        instance_counts = {obj: data['total_instances']
                           for obj, data in metrics.items()}

        figures.append(go.Figure(data=[
            go.Pie(labels=list(instance_counts.keys()),
                   values=list(instance_counts.values()),
                   name='Object Instances Distribution')
        ]))

        log_time("create_object_metrics_dashboard", "END", start)

        return figures

"""Modified create_ocpm_ui to generate OCEL files."""


def read_csv_with_detection(file):
    """Read CSV file with automatic separator detection"""
    # First, let's peek at the file content
    try:
        start = log_time("read_csv_with_detection", "START")
        # Read first few lines to detect the format
        file_content = file.read(1024).decode('utf-8')
        file.seek(0)  # Reset file pointer
        
        # Print first few lines for debugging
        print("First few lines of file:")
        print(file_content[:200])
        
        # Count potential separators
        separators = {
            ',': file_content.count(','),
            ';': file_content.count(';'),
            '\t': file_content.count('\t'),
            '|': file_content.count('|')
        }
        
        print("Detected separator counts:", separators)
        
        # Try separators in order of frequency
        for sep, count in sorted(separators.items(), key=lambda x: x[1], reverse=True):
            try:
                print(f"Trying separator: '{sep}'")
                df = pd.read_csv(file, sep=sep)
                print(f"Successfully read with separator: '{sep}'")
                print("Columns found:", df.columns.tolist())
                return df
            except Exception as e:
                print(f"Failed with separator '{sep}': {str(e)}")
                file.seek(0)  # Reset file pointer for next attempt
        
        # If no separator worked, try pandas' automatic detection
        print("Trying pandas automatic detection...")
        file.seek(0)
        df = pd.read_csv(file, engine='python')
        print("Successfully read with automatic detection")
        print("Columns found:", df.columns.tolist())
        log_time("read_csv_with_detection", "END", start)


        return df
        
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")


def create_ocpm_ui():      
    """Create OCPM analysis and generate OCEL file"""
    logger = logging.getLogger(__name__)

    # Check for existing event log
    event_log_path = "api_response/data.csv"

    try:
        start = log_time("create_ocpm_ui", "START")
        if os.path.exists(event_log_path):
            # Try reading with different separators
            try:
                # First try semicolon separator
                df = pd.read_csv(event_log_path, sep=';')
            except:
                # Fallback to comma separator
                df = pd.read_csv(event_log_path)

            logger.info("Performing APA Analysis...")

            # Debug information
            logger.debug("Available columns: %s", df.columns.tolist())

        else:
            logger.error("Please run Process Discovery first to analyze event log")
            return

        # Initialize analyzer
        analyzer = OCPMAnalyzer(df)

        # Generate OCEL file
        ocel_path = analyzer.save_ocel()
        logger.info(f"OCEL file generated: {ocel_path}")

        # Object Interactions
        logger.info("Analyzing Object Type Interactions")
        interactions = analyzer.analyze_object_interactions()
        logger.debug("Object Interactions: %s", interactions)

        # Object Metrics
        logger.info("Calculating Object Type Metrics")
        metrics = analyzer.calculate_object_metrics()
        logger.debug("Object Metrics: %s", metrics)

        # Object Lifecycles
        logger.info("Generating Object Lifecycles")
        for object_type in analyzer.object_types.keys():
            lifecycle_graph = analyzer.generate_object_lifecycle_graph(object_type)
            logger.debug(f"Lifecycle graph for {object_type}: {lifecycle_graph.edges()}")

        log_time("create_ocpm_ui", "END", start)

    except Exception as e:
        logger.error(f"Error in OCPM analysis: {str(e)}")
        if 'df' in locals():
            logger.debug("Available columns: %s", df.columns.tolist())
        logger.debug("Please ensure the event log has the required columns and data format")
