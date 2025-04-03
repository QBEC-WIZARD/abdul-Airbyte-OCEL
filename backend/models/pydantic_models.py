from pydantic import BaseModel,RootModel,Field
from typing import List
from typing import Dict,Any
from datetime import datetime
from pydantic.error_wrappers import ValidationError

# Pydantic model for CSV validation
class CSVResponse(BaseModel):
    message: str
    data: list[dict]

class BPMNResponse(BaseModel):
    success: bool
    status_code: int
    message: str
    bpmn_xml: str


class ImageResponse(BaseModel):
    success: bool
    status_code: int
    message: str
    image: str


class ActivityMetrics(BaseModel):
    total_instances: int
    total_activities: int
    avg_activities_per_instance: float
    top_activities: Dict[str, int]
    interaction_count: int

class MetricsContainer(BaseModel):
    metrics: Dict[str, ActivityMetrics]

class MetricModel(BaseModel):
    metrics: MetricsContainer

class LifecycleGraph(BaseModel):
    graph_dot: str

class LifecycleModel(BaseModel):
    lifecycle_graph: LifecycleGraph

class AIAnalysisResponse(BaseModel):
    total_events: int
    total_cases: int
    total_resources: int
    ai_analysis: str


class Transaction(BaseModel):
    at: str
    r: List[str]

class UserActivity(BaseModel):
    u: str
    act: List[Transaction]

class DataModel(BaseModel):
    data: List[UserActivity]

class InteractionsItem(BaseModel):
    elements:List[str]
    count:int

class InteractionsResponse(BaseModel):
    interactions:List[InteractionsItem]
    

class FMEAAnalysisResponse(BaseModel):
    failure_modes: int
    findings: str
    insights: str
    recommendations: str

class RPNDataPoint(BaseModel):
    rpn_value: int = Field(..., ge=0, description="Risk Priority Number value")
    count: int = Field(..., ge=0, description="Frequency of occurrence")

class RPNDistributionPlot(BaseModel):
    title: str = Field(..., description="Title of the RPN Distribution Plot")
    xlabel: str = Field(..., description="Label for the x-axis")
    ylabel: str = Field(..., description="Label for the y-axis")
    data: List[RPNDataPoint] = Field(..., description="List of RPN values and their corresponding counts")

class FailurePatterns(BaseModel):
    status: str
    message: str
    chart_data: Dict[str, Any]
    failure_counts: Dict[str, int]
    processed_failures: Dict[str, Any]
    explanation: Dict[str, Any]

class FailureLogic(BaseModel):
    try:
        failure_logic: FailurePatterns
        failure_markdown: str
    except ValidationError as e:
        print('start erroro')
        print(str(e)[:500])


class ResourceOutput(BaseModel):
    workload_data:List[Dict[str,Any]]
    resource_details:Dict[str,Dict[str,Dict]]
    explanation:Dict

class ResourceAnalysis(BaseModel):
    resource_outlier_data:ResourceOutput
    resource_outlier_markdown:str

class EventData(BaseModel):
    timestamp: datetime
    activity: str
    resource: str


class EventSequence(BaseModel):
    timestamp: datetime
    activity: str
    resource: str
    duration_from_start: float


class CaseMetrics(BaseModel):
    total_events: int
    activity_variety: int
    duration_hours: float


class CaseDetail(BaseModel):
    case_id: str
    warnings: List[str]
    metrics: CaseMetrics
    timeline_data: List[EventData]
    event_sequence: List[EventSequence]
    object_interactions: List[Any] = []


class CaseOverview(BaseModel):
    case_id: str
    z_score: float
    is_outlier: bool
    total_events: int
    activity_count: int


class CaseAnalysis(BaseModel):
    case_overview: List[CaseOverview]
    case_details: Dict[str, CaseDetail]


class CaseAnalysisDocument(BaseModel):
    case_analysis: CaseAnalysis
    case_analysis_markdown: str

class TimeDataTimeGap(BaseModel):
    time_data:Dict
    time_gaps:List[Dict]
    markdown:str

class TimeLogic(BaseModel):
    time_logic:TimeDataTimeGap