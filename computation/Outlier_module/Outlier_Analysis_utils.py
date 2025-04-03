import os
import pandas as pd
import networkx as nx
import pydot
import json
from computation.Outlier_module.ocpm_analysis import OCPMAnalyzer  # Assuming you have this module
# from ocpm_visualizer import OCPMVisualizer  # Assuming this exists
from typing import Dict
import logging
import numpy as np

EVENT_LOG_PATH = "api_response/data_event_log.csv"

def load_event_log():
    """Load event log from file."""
    if not os.path.exists(EVENT_LOG_PATH):
        raise FileNotFoundError("Event log not found. Run Process Discovery first.")

    try:
        df = pd.read_csv(EVENT_LOG_PATH, sep=";")
    except:
        df = pd.read_csv(EVENT_LOG_PATH)

    return df



from backend.MasterApi.Routers.central_log import log_time


def perform_analysis():
    start=log_time("perform_analysis_OUTLIER_UTILS","START")
    """Perform OCPM analysis and return processed data."""
    df = load_event_log()
    analyzer = OCPMAnalyzer(df)

    ocel_path = analyzer.save_ocel()
    log_time("perform_analysis_OUTLIER_UTILS","END",start)
    return {
        "ocel_path": ocel_path,
        "analyzer": analyzer
    } 

from computation.Outlier_module.IntegratedAPAAnalyzer import IntegratedAPAAnalyzer

def initialize_IntegratedAPAAnalyzer():
    try:
        analyzer = IntegratedAPAAnalyzer()
        try:
            ocel_path = os.path.join("api_response", "process_data.json")
            analyzer.load_ocel(ocel_path)
        except Exception as e:
            print(f'The error is as follows : {str(e)}')
        return analyzer
    except Exception as e:
        print(f'The error is the intitialization of IntegratedAPAAnalyzer is as follows {str(e)}')


def get_object_interactions(cache_OCPM):
    """Return object type interactions as a JSON response."""
    analysis = cache_OCPM
    start=log_time("get_object_interactions","START")
    interactions = analysis["analyzer"].analyze_object_interactions()
    log_time("get_object_interactions","END",start)
    return {"interactions": interactions}

def get_object_metrics(cache_OCPM):
    """Return object type metrics as a JSON response."""
    analysis = cache_OCPM
    start=log_time("get_object_metrics","START")
    metrics = analysis["analyzer"].calculate_object_metrics()
    log_time("get_object_metrics","END",start)
    return {"metrics": metrics}

def get_object_lifecycle_graph(cache_OCPM,object_type: str):
    """Generate object lifecycle graph and return as DOT format."""
    analysis = cache_OCPM
    start=log_time("get_object_lifecycle","START")
    lifecycle_graph = analysis["analyzer"].generate_object_lifecycle_graph(object_type)
    dot_graph = nx.nx_pydot.to_pydot(lifecycle_graph)
    log_time("get_object_lifecycle","END",start)
    return {"graph_dot": dot_graph.to_string()}

def _count_resource_interactions(self) -> int:
        """Count number of resource interactions in process"""
        interactions = 0
        for case_id, case_events in self.case_events.items():
            case_resources = set()
            for event_id in case_events:
                event = next((e for e in self.ocel_data['ocel:events'] if e['ocel:id'] == event_id), None)
                if event:
                    resource = event.get('ocel:attributes', {}).get('resource', 'Unknown')
                    case_resources.add(resource)
            if len(case_resources) > 1:
                interactions += len(case_resources) - 1
        return interactions

def _count_major_violations(self, duration_data: Dict) -> int:
        """Count number of major timing violations"""
        major_violations = 0
        for metrics in duration_data.values():
            if hasattr(metrics, 'details'):
                violations = metrics.details.get('outlier_events', {}).get('timing_gap', [])
                major_violations += sum(1 for v in violations
                                        if v['details'].get('time_gap_minutes', 0) >
                                        v['details'].get('threshold_minutes', 0) * 2)
        return major_violations

def _identify_bottlenecks(self, duration_data: Dict) -> int:
        """Identify number of process bottlenecks"""
        return sum(1 for metrics in duration_data.values()
                   if hasattr(metrics, 'is_outlier') and metrics.is_outlier)

def _count_object_interactions(self, case_data: Dict) -> int:
        """Count cases with multiple object interactions"""
        multi_object_cases = 0
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                events = metrics.details.get('events', {})
                if len(set(events.get('object_types', []))) > 1:
                    multi_object_cases += 1
        return multi_object_cases

def _count_case_variants(self, case_data: Dict) -> int:
        """Count unique case variants based on activity sequences"""
        variants = set()
        for metrics in case_data.values():
            if hasattr(metrics, 'details'):
                activity_sequence = tuple(metrics.details.get('events', {}).get('activity_sequence', []))
                if activity_sequence:
                    variants.add(activity_sequence)
        return len(variants)

def get_explanation(self, tab_type: str, metrics: Dict) -> Dict:
        """Generate AI explanation for OCEL analysis results"""
        logger.info(f"Generating OCEL explanation for: {tab_type}")

        try:
            context = self._build_analysis_context(tab_type, metrics)

            prompt = f"""
            Analyze this Object-Centric Process Mining data for {context['title']}:

            Process Context:
            {self._format_process_context()}

            Analysis Metrics:
            {self._format_metrics(context['metrics'])}

            Specific Patterns:
            {self._format_patterns(context['patterns'])}

            Object Type Interactions:
            {self._format_object_interactions()}

            Provide detailed analysis focusing on:
            1. Object-centric interactions and their impact on process performance
            2. Process conformance, deviations, and their root causes
            3. Resource and time utilization patterns affecting multiple object types
            4. Multi-perspective performance insights across object lifecycles
            5. Specific recommendations for process improvement

            Format response as:
            1. Key findings about object interactions and process patterns
            2. Critical insights about performance and conformance
            3. Specific, actionable recommendations for process improvement
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert in object-centric process mining analysis, focusing on multi-object interactions and process patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return self._format_ai_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "error": "Unable to generate explanation",
                "details": str(e)
            }

from computation.Outlier_module.Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer 

OCEL_PATH = "api_response/process_data.json"
# Create a function to initialize the UnfairOCELAnalyzer

def convert_numpy_types(data):
    if isinstance(data, np.bool_):  # Convert NumPy boolean to Python boolean
        return bool(data)
    elif isinstance(data, np.integer):  # Convert NumPy integer to Python int
        return int(data)
    elif isinstance(data, np.floating):  # Convert NumPy float to Python float
        return float(data)
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]  # Recursively convert lists
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}  # Recursively convert dicts
    return data  # Return original if no conversion needed

def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert to string
        return super().default(obj)

def intitialize_UnfairOCELAnalyzer():
    try:
        unfair_analyzer = UnfairOCELAnalyzer(OCEL_PATH)
        return unfair_analyzer
    except Exception as e:
        print(f'Intialization of OCEL failed due to {str(e)}')

def initialize_unfair_ocel_analyzer_with_failure_patterns(UnfairOCELAnalyzer_object):
    start=log_time("initialize_unfair_ocel_analyzer_with_failure_patterns","START")
    """Initialize UnfairOCELAnalyzer with OCEL data and process failure patterns."""
    unfair_analyzer = UnfairOCELAnalyzer_object
    markdown_failure_patterns = unfair_analyzer._display_failure_patterns_markdown()
    failure_patterns = unfair_analyzer.process_failure_patterns()
    
    #return (resouce_outlier_data)
    log_time("initialize_unfair_ocel_analyzer_with_failure_patterns","END",start)
    return {"failure_logic": failure_patterns,"failure_markdown": markdown_failure_patterns}


def initialize_unfair_ocel_analyzer_with_resource_analysis(UnfairOCELAnalyzer_object):
    unfair_analyzer = UnfairOCELAnalyzer_object
    resouce_outlier_data=unfair_analyzer.get_resource_outlier_data()
    resouce_outlier_data=convert_numpy_types(resouce_outlier_data)
    markdown_resource_outlier=unfair_analyzer._display_resource_outlier_markdown()
    return {"resource_outlier_data":resouce_outlier_data,"resource_outlier_markdown":markdown_resource_outlier}

def initialize_unfair_ocel_analyzer_with_time_analysis(UnfairOCELAnalyzer_object):
    """Initialize UnfairOCELAnalyzer with OCEL data and process time analysis."""
    try:
        start=log_time("initialize_unfair_ocel_analyzer_with_time_analysis","START")
        unfair_analyzer = UnfairOCELAnalyzer_object
        time_data_and_time_gaps = unfair_analyzer.analyze_time_outliers()
        markdown=unfair_analyzer.get_time_outlier_markdown()
        log_time("initialize_unfair_ocel_analyzer_with_time_analysis","END",start)
        return {"time_logic":time_data_and_time_gaps}
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the error as needed

def initialize_unfair_ocel_analyzer_with_case_analysis_patterns(UnfairOCELAnalyzer_object):
    """Initialize UnfairOCELAnalyzer with OCEL data and process case analysis."""
    try:
        unfair_analyzer = UnfairOCELAnalyzer_object
        case_analysis = unfair_analyzer.prepare_case_analysis()
        markdown=unfair_analyzer.display_case_outlier_markdown()

        case_analysis= convert_numpy_types(case_analysis)
        return {"case_analysis":case_analysis,"case_analysis_markdown":markdown}
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the error as needed
































def _display_failure_patterns_logic(self):
        """Display the logic content for Failure Patterns tab"""
        # Validate failures data exists
        failures_data = self.outliers.get('failures', {})
        if not failures_data:
            st.warning("No failure pattern data available")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                failure_counts = {
                    'Sequence Violations': len(failures_data.get('sequence_violations', [])),
                    'Incomplete Cases': len(failures_data.get('incomplete_cases', [])),
                    'Long Running': len(failures_data.get('long_running', [])),
                    'Resource Switches': len(failures_data.get('resource_switches', [])),
                    'Rework Activities': len(failures_data.get('rework_activities', []))
                }

                if any(failure_counts.values()):
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(failure_counts.keys()),
                            y=list(failure_counts.values()),
                            text=list(failure_counts.values()),
                            textposition='auto',
                        )
                    ])

                    fig.update_layout(
                        title='Process Failure Patterns Distribution',
                        xaxis_title='Failure Pattern Type',
                        yaxis_title='Count',
                        showlegend=False
                    )
                    st.plotly_chart(fig)

                    # Show failure details in expandable sections
                    for pattern, failures in failures_data.items():
                        if failures:
                            with st.expander(f"{pattern.replace('_', ' ').title()} ({len(failures)})"):
                                try:
                                    failure_df = pd.DataFrame(failures)
                                    st.dataframe(failure_df)
                                except Exception as e:
                                    logger.error(f"Error creating failure DataFrame: {str(e)}")
                                    st.error(f"Error displaying failure details: {str(e)}")

            with col2:
                failure_metrics = {
                    'sequence_violations': len(failures_data.get('sequence_violations', [])),
                    'incomplete_cases': len(failures_data.get('incomplete_cases', [])),
                    'long_running': len(failures_data.get('long_running', []))
                }

                st.markdown("### Understanding Failure Patterns")
                try:
                    explanation = self.get_explanation('failure', failure_metrics)
                    st.markdown(explanation.get('summary', 'No summary available'))

                    if explanation.get('insights'):
                        st.markdown("#### Key Insights")
                        for insight in explanation['insights']:
                            st.markdown(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.markdown("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.markdown(f"• {rec}")
                except Exception as e:
                    logger.error(f"Error getting explanation: {str(e)}")
                    st.error("Unable to generate insights at this time")


def _display_resource_outlier_logic(self):
        """Display the logic content for Resource Outlier tab"""
        # Validate resource data exists
        resource_data = self.outliers.get('resource_load', {})
        if not resource_data:
            st.warning("No resource analysis data available")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create resource workload visualization
                workload_data = []
                resource_details = {}

                for resource, metrics in resource_data.items():
                    if isinstance(metrics, (dict, object)):  # Validate metrics object
                        try:
                            workload_data.append({
                                'Resource': resource,
                                'Z-Score': getattr(metrics, 'z_score', 0),
                                'Is Outlier': getattr(metrics, 'is_outlier', False),
                                'Total Events': metrics.details['metrics'].get('total_events', 0) if hasattr(
                                    metrics, 'details') else 0,
                                'Cases': len(metrics.details['events'].get('cases', [])) if hasattr(metrics,
                                                                                                    'details') else 0,
                                'Activities': len(metrics.details['events'].get('activities', [])) if hasattr(
                                    metrics, 'details') else 0
                            })

                            # Store details for traceability
                            if hasattr(metrics, 'details'):
                                resource_details[resource] = {
                                    'metrics': metrics.details.get('metrics', {}),
                                    'events': metrics.details.get('events', {}),
                                    'patterns': metrics.details.get('outlier_patterns', {})
                                }
                        except Exception as e:
                            logger.error(f"Error processing resource metrics: {str(e)}")

                if workload_data:
                    # Create resource plot
                    try:
                        fig = px.scatter(
                            pd.DataFrame(workload_data),
                            x='Resource',
                            y='Z-Score',
                            size='Total Events',
                            color='Is Outlier',
                            title='Resource Workload Distribution',
                            custom_data=['Cases', 'Activities']
                        )

                        fig.update_traces(
                            hovertemplate=(
                                    "<b>%{x}</b><br>" +
                                    "Z-Score: %{y:.2f}<br>" +
                                    "Total Events: %{marker.size}<br>" +
                                    "Cases: %{customdata[0]}<br>" +
                                    "Activities: %{customdata[1]}<br>" +
                                    "<extra></extra>"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show resource details
                        if resource_details:
                            selected_resource = st.selectbox(
                                "Select resource for details",
                                options=list(resource_details.keys())
                            )
                            if selected_resource:
                                self._display_resource_details(selected_resource,
                                                               resource_details[selected_resource])
                    except Exception as e:
                        logger.error(f"Error creating resource visualization: {str(e)}")
                        st.error("Error displaying resource visualization")

            with col2:
                workload_metrics = {
                    'market_maker_b': len(self.resource_events.get('Market Maker B', [])),
                    'client_desk_d': len(self.resource_events.get('Client Desk D', [])),
                    'options_desk_a': len(self.resource_events.get('Options Desk A', []))
                }
                st.markdown("### Understanding Resource Distribution")
                try:
                    explanation = self.get_explanation('resource', workload_metrics)
                    st.markdown(explanation.get('summary', 'No summary available'))

                    if explanation.get('insights'):
                        st.markdown("#### Key Insights")
                        for insight in explanation['insights']:
                            st.markdown(f"• {insight}")

                    if explanation.get('recommendations'):
                        st.markdown("#### Recommendations")
                        for rec in explanation['recommendations']:
                            st.markdown(f"• {rec}")
                except Exception as e:
                    logger.error(f"Error getting resource explanation: {str(e)}")
                    st.error("Unable to generate resource insights")

def display_time_outlier_tab(self):
        """Display the Time Outlier tab content."""
        logger.debug("Processing Time Analysis tab")

        # Validate duration data exists
        duration_data = self.outliers.get('duration', {})
        if not duration_data:
            st.warning("No time analysis data available")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            try:
                time_data=self._analyze_time_data(duration_data) # return dictionay with pd data frame contains
                time_gaps=self._analyze_timing_gaps(duration_data) # return pandas dataframe
            except Exception as e:
                logger.error(f"Error displaying time analysis: {str(e)}")
                st.error("Error displaying time analysis visualizations")

        with col2:
            timing_metrics = {
                'avg_duration': f"{np.mean([len(events) for events in self.case_events.values()]):.2f}",
                'outlier_count': len(
                    [m for m in duration_data.values() if getattr(m, 'is_outlier', False)]),
                'typical_duration': "2-4 hours"
            }
            st.markdown("### Understanding Time Patterns")

            try:
                explanation = self.get_explanation('time', timing_metrics)
                st.write(explanation.get('summary', 'No summary available'))

                if explanation.get('insights'):
                    st.markdown("#### Key Insights")
                    for insight in explanation['insights']:
                        st.markdown(f"• {insight}")

                if explanation.get('recommendations'):
                    st.markdown("#### Recommendations")
                    for rec in explanation['recommendations']:
                        st.markdown(f"• {rec}")
            except Exception as e:
                logger.error(f"Error getting time explanation: {str(e)}")
                st.error("Unable to generate time insights")

def display_enhanced_analysis(self):
        """Display enhanced analysis with comprehensive outlier tracing and error handling"""
        try:
            # Validate that outliers dictionary exists and has required keys
            if not hasattr(self, 'outliers'):
                st.error("No outlier analysis data available. Please ensure data is processed first.")
                return

            # Create tabs for different analyses
            tabs = st.tabs(["Failure Patterns", "Resource Outlier", "Time Outlier", "Case Outlier"])

            # Failure Patterns Tab
            with tabs[0]:
                logger.debug("Processing Failure Patterns tab")
                with st.expander("Failure Pattern Detection Logic"):
                    st.markdown("""
                    # Understanding Failure Pattern Detection

                    ## Overview
                    The failure pattern detection system analyzes object-centric event logs to identify various types of process deviations and anomalies. It monitors six key categories of failures:
                    
                    1. Sequence Violations
                    2. Incomplete Cases
                    3. Long Running Cases
                    4. Resource Switches  
                    5. Rework Activities
                    6. Timing Violations
                    
                    ## Detection Process
                    
                    ### Data Preparation
                    The system first organizes event data into a structured format containing:
                    - Event IDs
                    - Timestamps
                    - Activities
                    - Resources
                    - Object Types
                    - Object IDs
                    
                    ### Failure Categories in Detail
                    
                    #### 1. Sequence Violations
                    ```python
                    actual_sequence = case_data[...]['activity'].tolist()
                    if actual_sequence != expected_sequence:
                        # Track violation details
                    ```
                    - Compares actual activity sequence against expected flow
                    - Records:
                      - Missing activities
                      - Wrong order activities
                      - First violation point
                      - Affected objects
                    
                    #### 2. Incomplete Cases
                    ```python
                    if violation['missing_activities']:
                        # Track incomplete case details
                    ```
                    - Identifies cases missing required activities
                    - Tracks:
                      - Missing activities list
                      - Completed activities
                      - Last known event
                      - Case context
                    
                    #### 3. Timing Violations
                    ```python
                    if case_duration > timing_rules['total_duration']:
                        # Track timing violation details
                    ```
                    Monitors two types of timing issues:
                    - Overall case duration exceeding thresholds
                    - Activity-specific gaps between events
                    - Records detailed gap analysis including:
                      - Previous activity
                      - Current activity
                      - Gap duration
                      - Threshold exceeded
                    
                    #### 4. Resource Switches
                    ```python
                    resource_changes = [(i, sequence[i], sequence[i+1])
                                       for i in range(len(sequence)-1)
                                       if sequence[i] != sequence[i+1]]
                    ```
                    - Detects handovers between different resources
                    - Tracks:
                      - Switch points
                      - From/To resources
                      - Associated activities
                      - Timestamps
                    
                    #### 5. Rework Activities
                    ```python
                    activity_counts = case_data['activity'].value_counts()
                    rework = activity_counts[activity_counts > 1]
                    ```
                    - Identifies repeated activities
                    - Records:
                      - Activity frequency
                      - Event sequences
                      - Resources involved
                      - Timestamps
                    
                    ## Visualization Components
                    
                    The failure pattern analysis is displayed in four key components:
                    
                    1. **Pattern Distribution Bar Chart**
                       - Shows count of each failure type
                       - Color-coded by severity
                       - Interactive tooltips with details
                    
                    2. **Detailed Pattern Analysis**
                       - Expandable sections for each pattern type
                       - Tabular view of specific failures
                       - Sorting and filtering capabilities
                    
                    3. **Metrics Summary**
                       - Key statistics about detected patterns
                       - Trend indicators
                       - Severity distribution
                    
                    4. **AI-Generated Insights**
                       - Pattern interpretation
                       - Key findings
                       - Improvement recommendations
                    
                    ## Usage in Process Analysis
                    
                    This failure pattern detection helps organizations:
                    1. Identify process bottlenecks
                    2. Monitor compliance violations
                    3. Optimize resource allocation
                    4. Improve process efficiency
                    5. Ensure quality control
                    
                    ## Implementation Details
                    
                    The code implements several advanced features:
                    - Full event traceability
                    - Multi-perspective analysis
                    - Object-centric correlation
                    - Temporal pattern detection
                    - Resource interaction analysis
                    
                    ## Data Structure Example
                    
                    A typical failure pattern record looks like:
                    ```json
                    {
                      "case_id": "Case_1",
                      "object_type": "Trade",
                      "actual_sequence": ["A", "B", "D"],
                      "expected_sequence": ["A", "B", "C", "D"],
                      "missing_activities": ["C"],
                      "events": ["evt_1", "evt_2", "evt_4"],
                      "first_violation": {
                        "event_id": "evt_2",
                        "timestamp": "2024-01-01T10:00:00",
                        "resource": "Trader_A"
                      }
                    }
                    ```
                    
                    ## Performance Considerations
                    
                    The detection system is optimized for:
                    - Efficient data processing
                    - Minimal memory footprint
                    - Real-time analysis capability
                    - Scalable pattern detection
                    
                    ## Error Handling
                    
                    The system includes comprehensive error handling:
                    - Data validation
                    - Exception logging
                    - Graceful degradation
                    - Recovery mechanisms
                    
                    ## Integration Points
                    
                    The failure pattern detection integrates with:
                    - Process mining analytics
                    - Conformance checking
                    - Performance analysis
                    - Resource optimization
                    
                    
                    """)

                # Validate failures data exists
                failures_data = self.outliers.get('failures', {})
                if not failures_data:
                    st.warning("No failure pattern data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        failure_counts = {
                            'Sequence Violations': len(failures_data.get('sequence_violations', [])),
                            'Incomplete Cases': len(failures_data.get('incomplete_cases', [])),
                            'Long Running': len(failures_data.get('long_running', [])),
                            'Resource Switches': len(failures_data.get('resource_switches', [])),
                            'Rework Activities': len(failures_data.get('rework_activities', []))
                        }

                        if any(failure_counts.values()):
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(failure_counts.keys()),
                                    y=list(failure_counts.values()),
                                    text=list(failure_counts.values()),
                                    textposition='auto',
                                )
                            ])

                            fig.update_layout(
                                title='Process Failure Patterns Distribution',
                                xaxis_title='Failure Pattern Type',
                                yaxis_title='Count',
                                showlegend=False
                            )
                            st.plotly_chart(fig)

                            # Show failure details in expandable sections
                            for pattern, failures in failures_data.items():
                                if failures:
                                    with st.expander(f"{pattern.replace('_', ' ').title()} ({len(failures)})"):
                                        try:
                                            failure_df = pd.DataFrame(failures)
                                            st.dataframe(failure_df)
                                        except Exception as e:
                                            logger.error(f"Error creating failure DataFrame: {str(e)}")
                                            st.error(f"Error displaying failure details: {str(e)}")

                    with col2:
                        failure_metrics = {
                            'sequence_violations': len(failures_data.get('sequence_violations', [])),
                            'incomplete_cases': len(failures_data.get('incomplete_cases', [])),
                            'long_running': len(failures_data.get('long_running', []))
                        }

                        st.markdown("### Understanding Failure Patterns")
                        try:
                            explanation = self.get_explanation('failure', failure_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting explanation: {str(e)}")
                            st.error("Unable to generate insights at this time")

            # Resource Analysis Tab
            with tabs[1]:
                logger.debug("Processing Resource Analysis tab")
                with st.expander("📊 Resource Complexity Detection Logic"):
                    st.markdown("""
                    # Understanding Resource Complexity Detection

                    ## Overview
                    The visualization shows resource workload distribution in object-centric process mining (OCPM), highlighting potential outliers in how resources interact with different process objects and activities.
                    
                    ## Metrics Calculation
                    
                    ### Base Metrics
                    - **Total Events**: Raw count of events handled by each resource
                    - **Unique Cases**: Number of distinct cases a resource works on
                    - **Activity Variety**: Number of different activities performed
                    - **Object Variety**: Unique object types handled
                    
                    ### Z-Score Analysis
                    The code calculates normalized z-scores for each metric using:
                    ```
                    z = (value - mean) / standard_deviation
                    ```
                    
                    ### Composite Score
                    A composite z-score is generated by averaging absolute z-scores across all metrics to identify overall outliers.
                    
                    ## Visualization Components
                    
                    ### Scatter Plot Elements
                    - **X-axis**: Individual resources
                    - **Y-axis**: Composite z-score
                    - **Bubble Size**: Total event count
                    - **Color**: Outlier status (z-score > 3)
                    
                    ### Interactive Features
                    - Hover shows detailed metrics
                    - Selection enables detailed resource analysis
                    
                    ## Outlier Detection
                    
                    ### Workload Patterns
                    - **High Workload**: Events > mean + 2*std
                    - **High Variety**: Activity count > mean + 2*std
                    
                    ### Resource Classification
                    Resources are flagged as outliers when:
                    - Composite z-score > 3
                    - Showing unusual patterns in:
                      - Event volume
                      - Case variety
                      - Activity diversity
                      - Object type interactions
                    
                    ## Interpretation Guide
                    
                    ### Normal Resource Profile
                    - Balanced workload distribution
                    - Expected activity variety
                    - Typical object type interactions
                    
                    ### Outlier Indicators
                    - Unusually high/low event counts
                    - Excessive activity variety
                    - Abnormal object type interactions
                    - Extreme composite z-scores
                    
                    ### Business Impact
                    - Resource overloading
                    - Specialization vs. generalization
                    - Process bottlenecks
                    - Workload imbalances
                    
                    ## Technical Implementation Notes
                    The implementation uses vectorized operations in pandas for efficiency:
                    - Grouped aggregations
                    - Vectorized z-score calculations
                    - Optimized outlier detection
                    
                    """)

                    # Validate resource data exists
                resource_data = self.outliers.get('resource_load', {})
                if not resource_data:
                    st.warning("No resource analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Create resource workload visualization
                        workload_data = []
                        resource_details = {}

                        for resource, metrics in resource_data.items():
                            if isinstance(metrics, (dict, object)):  # Validate metrics object
                                try:
                                    workload_data.append({
                                        'Resource': resource,
                                        'Z-Score': getattr(metrics, 'z_score', 0),
                                        'Is Outlier': getattr(metrics, 'is_outlier', False),
                                        'Total Events': metrics.details['metrics'].get('total_events', 0) if hasattr(
                                            metrics, 'details') else 0,
                                        'Cases': len(metrics.details['events'].get('cases', [])) if hasattr(metrics,
                                                                                                            'details') else 0,
                                        'Activities': len(metrics.details['events'].get('activities', [])) if hasattr(
                                            metrics, 'details') else 0
                                    })

                                    # Store details for traceability
                                    if hasattr(metrics, 'details'):
                                        resource_details[resource] = {
                                            'metrics': metrics.details.get('metrics', {}),
                                            'events': metrics.details.get('events', {}),
                                            'patterns': metrics.details.get('outlier_patterns', {})
                                        }
                                except Exception as e:
                                    logger.error(f"Error processing resource metrics: {str(e)}")

                        if workload_data:
                            # Create resource plot
                            try:
                                fig = px.scatter(
                                    pd.DataFrame(workload_data),
                                    x='Resource',
                                    y='Z-Score',
                                    size='Total Events',
                                    color='Is Outlier',
                                    title='Resource Workload Distribution',
                                    custom_data=['Cases', 'Activities']
                                )

                                fig.update_traces(
                                    hovertemplate=(
                                            "<b>%{x}</b><br>" +
                                            "Z-Score: %{y:.2f}<br>" +
                                            "Total Events: %{marker.size}<br>" +
                                            "Cases: %{customdata[0]}<br>" +
                                            "Activities: %{customdata[1]}<br>" +
                                            "<extra></extra>"
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Show resource details
                                if resource_details:
                                    selected_resource = st.selectbox(
                                        "Select resource for details",
                                        options=list(resource_details.keys())
                                    )
                                    if selected_resource:
                                        self._display_resource_details(selected_resource,
                                                                       resource_details[selected_resource])
                            except Exception as e:
                                logger.error(f"Error creating resource visualization: {str(e)}")
                                st.error("Error displaying resource visualization")

                    with col2:
                        workload_metrics = {
                            'market_maker_b': len(self.resource_events.get('Market Maker B', [])),
                            'client_desk_d': len(self.resource_events.get('Client Desk D', [])),
                            'options_desk_a': len(self.resource_events.get('Options Desk A', []))
                        }
                        st.markdown("### Understanding Resource Distribution")
                        try:
                            explanation = self.get_explanation('resource', workload_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting resource explanation: {str(e)}")
                            st.error("Unable to generate resource insights")

            # Time Analysis Tab
            with tabs[2]:
                logger.debug("Processing Time Analysis tab")
                # Under the Time Analysis Tab, after your visualizations
                with st.expander("🔍 Time Outlier Detection Logic"):
                    st.markdown("""
                    # Object-Centric Process Mining: Time Outlier Detection

                    ## Overview
                    The time outlier detection analyzes process execution durations and timing patterns across different object types and activities. The visualization appears in the "Time Outlier" tab and consists of two key components:
                    
                    1. Duration Distribution Plot
                    2. Timing Gap Analysis Table
                    
                    ## Detection Process
                    
                    ### 1. Duration Data Collection
                    The system:
                    - Creates a DataFrame of all events with timestamps, activities, and object relationships
                    - Groups events by activity to analyze timing patterns
                    - Tracks multiple object types per event to handle object-centric complexity
                    
                    ### 2. Threshold Validation
                    For each activity-object combination:
                    - Retrieves timing rules from OCPM validator
                    - Validates against:
                      - Activity-specific thresholds
                      - Default gap thresholds per object type
                      - Overall process duration limits
                    
                    ### 3. Outlier Detection Logic
                    
                    #### Duration Outliers
                    ```python
                    gap_hours = (event['timestamp'] - last_event['timestamp']).total_seconds() / 3600
                    if gap_hours > activity_threshold:
                        outlier_events['timing_gap'].append({...})
                    ```
                    
                    System flags outliers when:
                    - Time gap between activities exceeds threshold
                    - Events occur out of expected sequence
                    - Activities take longer than typical duration
                    
                    #### Z-Score Calculation
                    ```python
                    violation_score = sum(metrics.values()) / (metrics['total_events'] * 3)
                    z_score = float(violation_score * 10)
                    is_outlier = violation_score > 0.3
                    ```
                    
                    ### 4. Visualization Components
                    
                    #### Duration Distribution Plot
                    - X-axis: Activities
                    - Y-axis: Z-Score
                    - Size: Number of events
                    - Color: Outlier status (True/False)
                    
                    #### Timing Gap Table
                    Shows detailed timing violations:
                    - Case ID
                    - Current/Previous Activities
                    - Gap Duration
                    - Threshold Values
                    - Color coding:
                      - Red: Exceeds threshold
                      - Green: Within threshold
                    
                    ## Key Metrics Tracked
                    
                    1. **Timing Violations**
                       - Gap between activities
                       - Sequence position violations
                       - Resource handover delays
                    
                    2. **Activity Statistics**
                       - Average duration per activity
                       - Resource distribution
                       - Object type distribution
                    
                    3. **Outlier Metrics**
                       - Z-score per activity
                       - Violation rate
                       - Total events and violations
                    
                    ## Example Interpretation
                    
                    If an activity shows:
                    - Z-score > 3: Significant outlier
                    - Multiple timing gaps: Process bottleneck
                    - High violation rate: Potential process issue
                    
                    ## Understanding the Visualization
                    
                    1. **Scatter Plot Reading**
                       - Each point represents an activity
                       - Size indicates event frequency
                       - Higher Z-scores suggest more severe timing issues
                       - Color differentiates outliers from normal activities
                    
                    2. **Gap Analysis Table Reading**
                       - Red rows indicate critical timing violations
                       - Compare actual gaps against thresholds
                       - Look for patterns in specific cases or activities
                    
                    ## Technical Implementation Notes
                    
                    The detection uses a multi-level approach:
                    1. Event-level timing analysis
                    2. Activity-level pattern detection
                    3. Object-centric relationship validation
                    4. Cross-object timing correlation
                    
                    This ensures comprehensive coverage of timing patterns while maintaining object-centric process mining principles.
                    """)

                # Validate duration data exists
                duration_data = self.outliers.get('duration', {})
                if not duration_data:
                    st.warning("No time analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        try:
                            self._display_time_analysis(duration_data)
                        except Exception as e:
                            logger.error(f"Error displaying time analysis: {str(e)}")
                            st.error("Error displaying time analysis visualizations")

                    with col2:
                        timing_metrics = {
                            'avg_duration': f"{np.mean([len(events) for events in self.case_events.values()]):.2f}",
                            'outlier_count': len(
                                [m for m in duration_data.values() if getattr(m, 'is_outlier', False)]),
                            'typical_duration': "2-4 hours"
                        }
                        st.markdown("### Understanding Time Patterns")
                        try:
                            explanation = self.get_explanation('time', timing_metrics)
                            st.write(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting time explanation: {str(e)}")
                            st.error("Unable to generate time insights")

            # Case Analysis Tab
            with tabs[3]:
                logger.debug("Processing Case Analysis tab")

                with st.expander("Case Outlier Detection Logic", expanded=False):
                    st.markdown("""
                    ## Case Outlier Detection in Object-Centric Process Mining

                    ### 1. Data Structure & Analysis
                    The code analyzes case outliers using these key metrics:
                    - **Total Events**: Number of events per case
                    - **Activity Variety**: Unique activities in each case
                    - **Resource Variety**: Different resources involved
                    - **Object Type Variety**: Different object types per case
                    - **Case Duration**: Total duration in hours

                    ### 2. Detection Method
                    Cases are flagged as outliers based on:
                    - Composite z-score calculation across all metrics
                    - Threshold: z-score > 3 indicates outlier
                    - Multi-dimensional analysis including:
                        - Event frequency patterns
                        - Activity sequence variations
                        - Resource utilization patterns
                        - Object type interactions

                    ### 3. Visualization Components

                    #### Main Scatter Plot
                    - X-axis: Case IDs
                    - Y-axis: Z-scores
                    - Point Size: Total events in case
                    - Color: Outlier status (True/False)
                    - Hover data: Detailed case metrics

                    #### Case Details View
                    When selecting a specific case:
                    - Event sequence timeline
                    - Resource distribution
                    - Object type interactions
                    - Duration metrics

                    ### 4. Implementation Details
                    ```python
                    # Key method calls in order:
                    1. _detect_case_outliers()
                    2. _build_trace_index()
                    3. _display_case_analysis()
                    4. _display_case_details()
                    ```

                    ### 5. Outlier Classification
                    Cases are marked as outliers if they exhibit:
                    - Unusually high/low number of events
                    - Unexpected activity patterns
                    - Abnormal resource usage
                    - Complex object interactions
                    - Extreme duration values

                    ### 6. AI Enhancement
                    The analysis includes AI-powered insights:
                    - Pattern identification
                    - Root cause analysis
                    - Improvement recommendations
                    """)

                    st.info("""
                    **How to Use the Visualization:**
                    1. Examine the scatter plot for overall outlier distribution
                    2. Click on specific cases to view detailed analysis
                    3. Review AI insights for process understanding
                    4. Check object interactions for complexity analysis
                    """)

                # Validate case complexity data exists
                case_data = self.outliers.get('case_complexity', {})
                if not case_data:
                    st.warning("No case analysis data available")
                else:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        try:
                            self._display_case_analysis(case_data)
                        except Exception as e:
                            logger.error(f"Error displaying case analysis: {str(e)}")
                            st.error("Error displaying case analysis visualizations")

                    with col2:
                        case_metrics = {
                            'complex_cases': len([c for c in case_data.values()
                                                  if c.details['metrics'].get('total_events', 0) > 10]),
                            'timestamp_cases': len([c for c in case_data.values()
                                                    if c.details['temporal'].get('duration_hours', 0) > 24]),
                            'object_cases': len(case_data)
                        }
                        st.markdown("### Understanding Case Complexity")
                        try:
                            explanation = self.get_explanation('case', case_metrics)
                            st.markdown(explanation.get('summary', 'No summary available'))

                            if explanation.get('insights'):
                                st.markdown("#### Key Insights")
                                for insight in explanation['insights']:
                                    st.markdown(f"• {insight}")

                            if explanation.get('recommendations'):
                                st.markdown("#### Recommendations")
                                for rec in explanation['recommendations']:
                                    st.markdown(f"• {rec}")
                        except Exception as e:
                            logger.error(f"Error getting case explanation: {str(e)}")
                            st.error("Unable to generate case insights")

        except Exception as e:
            logger.error(f"Error in display_enhanced_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("Error analyzing process patterns. Please check logs for details.")
            if st.checkbox("Show detailed error"):
                st.code(traceback.format_exc())


def _display_timing_gaps(self, duration_data: Dict):
        """Helper method to display timing gap analysis using pandas styling"""
        try:
            timing_gaps = []
            for activity, metrics in duration_data.items():
                if hasattr(metrics, 'details'):
                    for violation in metrics.details.get('outlier_events', {}).get('timing_gap', []):
                        prev_event = self._get_event_details(violation['details'].get('previous_event', ''))

                        timing_gaps.append({
                            'Case ID': violation.get('case_id', 'Unknown'),
                            'Current Activity': activity,
                            'Previous Activity': prev_event.get('Activity', 'Unknown'),
                            'Gap (Minutes)': round(violation['details'].get('time_gap_minutes', 0), 2),
                            'Threshold': violation['details'].get('threshold_minutes', 0)
                        })

            if timing_gaps:
                st.write("### Timing Gap Analysis")
                df = pd.DataFrame(timing_gaps)

                def highlight_gaps(row):
                    return ['background-color: #ffcdd2' if row['Gap (Minutes)'] > row['Threshold']
                            else 'background-color: #ffffff' for _ in row]

                styled_df = df.style.apply(highlight_gaps, axis=1)
                st.dataframe(styled_df)

        except Exception as e:
            logger.error(f"Error displaying timing gaps: {str(e)}")
            st.error("Error displaying timing gap analysis")


def _display_case_details(self, case_id: str, details: Dict):
        """Helper method to display detailed case information"""
        try:
            st.write("### Case Details")

            # Show if case is an outlier
            if details.get('outlier_patterns', {}).get('high_complexity'):
                st.warning("⚠️ This case has high complexity")
            if details.get('outlier_patterns', {}).get('long_duration'):
                st.warning("⚠️ This case has unusual duration")

            # Display metrics
            cols = st.columns(3)
            cols[0].metric(
                "Total Events",
                details['metrics'].get('total_events', 0)
            )
            cols[1].metric(
                "Activities",
                details['metrics'].get('activity_variety', 0)
            )
            cols[2].metric(
                "Duration (hrs)",
                f"{details['temporal'].get('duration_hours', 0):.1f}"
            )

            # Display timeline
            events_df = self._get_events_dataframe(details['events'].get('all_events', []))
            if not events_df.empty:
                events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

                # Create timeline visualization
                timeline_fig = px.scatter(
                    events_df,
                    x='timestamp',
                    y='activity',
                    color='resource',
                    title=f'Case Timeline - {case_id}',
                    labels={
                        'timestamp': 'Time',
                        'activity': 'Activity',
                        'resource': 'Resource'
                    }
                )

                timeline_fig.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='closest'
                )
                st.plotly_chart(timeline_fig)

                # Display event sequence
                st.write("### Event Sequence")
                sequence_df = events_df[['timestamp', 'activity', 'resource']].copy()
                sequence_df['duration_from_start'] = (
                                                             sequence_df['timestamp'] - sequence_df['timestamp'].min()
                                                     ).dt.total_seconds() / 3600

                # Add styling to highlight potential issues
                styled_df = sequence_df.style.background_gradient(
                    subset=['duration_from_start'],
                    cmap='YlOrRd'
                )
                st.dataframe(styled_df)

            # Display object interactions if available
            if 'object_types' in details:
                st.write("### Object Interactions")
                object_df = pd.DataFrame({
                    'Object Type': details['object_types'],
                    'Interaction Count': [1] * len(details['object_types'])
                }).groupby('Object Type').sum()

                st.bar_chart(object_df)

        except Exception as e:
            logger.error(f"Error displaying case details: {str(e)}")
            st.error(f"Error showing case details: {str(e)}")

def _display_case_analysis(self, case_data: Dict):
        """Helper method to display case analysis"""
        try:
            # Create case visualization
            case_info = []
            case_details = {}

            for case_id, metrics in case_data.items():
                if hasattr(metrics, 'details'):
                    case_info.append({
                        'Case': case_id,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier,
                        'Total Events': metrics.details['metrics'].get('total_events', 0),
                        'Activity Count': metrics.details['metrics'].get('activity_variety', 0)
                    })
                    case_details[case_id] = metrics.details

            if case_info:
                df = pd.DataFrame(case_info)
                fig = px.scatter(
                    df,
                    x='Case',
                    y='Z-Score',
                    size='Total Events',
                    color='Is Outlier',
                    title='Case Complexity Distribution'
                )

                fig.update_traces(
                    hovertemplate=(
                            "<b>%{x}</b><br>" +
                            "Z-Score: %{y:.2f}<br>" +
                            "Total Events: %{marker.size}<br>" +
                            "Activities: %{customdata[0]}<br>" +
                            "<extra></extra>"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add case details selection
                if case_details:
                    selected_case = st.selectbox(
                        "Select case for details",
                        options=list(case_details.keys())
                    )
                    if selected_case:
                        self._display_case_details(selected_case, case_details[selected_case])

        except Exception as e:
            logger.error(f"Error in case analysis display: {str(e)}")
            raise

def _display_time_analysis(self, duration_data: Dict):
        """Helper method to display time analysis"""
        try:
            # Create duration outlier visualization
            duration_info = []
            for activity, metrics in duration_data.items():
                if hasattr(metrics, 'details'):
                    duration_info.append({
                        'Activity': activity,
                        'Z-Score': metrics.z_score,
                        'Is Outlier': metrics.is_outlier,
                        'Total Events': metrics.details.get('total_events', 0),
                        'Violation Count': len(metrics.details.get('outlier_events', {}).get('timing_gap', []))
                    })

            if duration_info:
                df = pd.DataFrame(duration_info)
                fig = px.scatter(
                    df,
                    x='Activity',
                    y='Z-Score',
                    size='Total Events',
                    color='Is Outlier',
                    title='Activity Duration Distribution'
                )
                st.plotly_chart(fig)

                # Display timing gaps if available
                # self._analyze_timing_gaps(duration_data)

        except Exception as e:
            logger.error(f"Error in time analysis display: {str(e)}")
            raise
        
def case_outlier_logic(self):
            # Validate case complexity data exists
            case_data = self.outliers.get('case_complexity', {})
            if not case_data:
                st.warning("No case analysis data available")
            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    try:
                        self._display_case_analysis(case_data)
                    except Exception as e:
                        logger.error(f"Error displaying case analysis: {str(e)}")
                        st.error("Error displaying case analysis visualizations")

                with col2:
                    case_metrics = {
                        'complex_cases': len([c for c in case_data.values()
                                            if c.details['metrics'].get('total_events', 0) > 10]),
                        'timestamp_cases': len([c for c in case_data.values()
                                                if c.details['temporal'].get('duration_hours', 0) > 24]),
                        'object_cases': len(case_data)
                    }
                    st.markdown("### Understanding Case Complexity")
                    try:
                        explanation = self.get_explanation('case', case_metrics)
                        st.markdown(explanation.get('summary', 'No summary available'))

                        if explanation.get('insights'):
                            st.markdown("#### Key Insights")
                            for insight in explanation['insights']:
                                st.markdown(f"• {insight}")

                        if explanation.get('recommendations'):
                            st.markdown("#### Recommendations")
                            for rec in explanation['recommendations']:
                                st.markdown(f"• {rec}")
                    except Exception as e:
                        logger.error(f"Error getting case explanation: {str(e)}")
                        st.error("Unable to generate case insights")


def _calculate_time_metrics(self):
        """Calculate time-based metrics"""
        return {
            'activity_durations': self.relationships_df.groupby('activity')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

def _calculate_case_metrics(self):
        """Calculate case-based metrics"""
        return {
            'complexity': self.relationships_df.groupby('case_id')['activity'].nunique(),
            'duration': self.relationships_df.groupby('case_id')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }

def _calculate_handover_metrics(self):
        """Calculate handover metrics between resources"""
        return {
            'handovers': self.relationships_df.groupby('case_id')['resource'].agg(list).apply(
                lambda x: len([i for i in range(len(x) - 1) if x[i] != x[i + 1]])
            )
        }

def _create_resource_plot(self, metrics):
        """Create resource discrimination plot"""
        fig, ax = plt.subplots()
        metrics['workload'].plot(kind='bar', ax=ax)
        ax.set_title('Resource Workload Distribution')
        return fig

def _create_time_plot(self, metrics):
        """Create time bias plot"""
        fig, ax = plt.subplots()
        metrics['activity_durations'].plot(kind='bar', ax=ax)
        ax.set_title('Activity Duration Distribution')
        return fig

def _create_case_plot(self, metrics):
        """Create case priority plot"""
        fig, ax = plt.subplots()
        metrics['complexity'].plot(kind='hist', ax=ax)
        ax.set_title('Case Complexity Distribution')
        return fig

def _create_handover_plot(self, metrics):
        """Create handover analysis plot"""
        fig, ax = plt.subplots()
        metrics['handovers'].plot(kind='hist', ax=ax)
        ax.set_title('Handover Distribution')
        return fig

def _calculate_resource_metrics(self):
        """Calculate resource metrics"""
        return {
            'workload': self.relationships_df.groupby('resource').size(),
            'avg_duration': self.relationships_df.groupby('resource')['timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds() / 3600
            )
        }