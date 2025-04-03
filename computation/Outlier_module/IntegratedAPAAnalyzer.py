
import json
import pandas as pd

from utils import get_azure_openai_client

# import px
import plotly.express as px


from backend.MasterApi.Routers.central_log import log_time


class IntegratedAPAAnalyzer:
    """Integrated APA Analytics with AI Analysis capabilities"""

    def __init__(self, api_key: str = None):
        # Initialize OpenAI client with working configuration from test file
        self.client = get_azure_openai_client()
        self.ocel_data = None
        self.events_df = None
        self.stats = {}

    def load_ocel(self, file_path: str) -> None:
        """Load and process OCEL data"""
        try:
            start=log_time("load_ocel","START")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.ocel_data = json.load(f)

            events = []
            for event in self.ocel_data['ocel:events']:
                event_data = {
                    'event_id': event['ocel:id'],
                    'timestamp': pd.to_datetime(event['ocel:timestamp']),
                    'activity': event['ocel:activity'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                    'object_type': event.get('ocel:attributes', {}).get('object_type', 'Unknown'),
                    'objects': [obj['id'] for obj in event.get('ocel:objects', [])]
                }
                events.append(event_data)

            self.events_df = pd.DataFrame(events)
            self._calculate_statistics()
            log_time("load_ocel","END",start)
        except Exception as e:
            raise Exception(f"Error loading OCEL file: {str(e)}")

    def _calculate_statistics(self) -> None:
        """Calculate OCEL statistics"""
        if self.events_df is None:
            return

        self.stats = {
            'general': {
                'total_events': len(self.events_df),
                'total_cases': self.events_df['case_id'].nunique(),
                'date_range': {
                    'start': self.events_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': self.events_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'total_resources': self.events_df['resource'].nunique()
            },
            'activities': {
                'unique_count': self.events_df['activity'].nunique(),
                'distribution': self.events_df['activity'].value_counts().to_dict()
            },
            'resources': {
                'distribution': self.events_df['resource'].value_counts().to_dict()
            },
            'object_types': {
                'count': len(self.ocel_data.get('ocel:object-types', [])),
                'types': self.ocel_data.get('ocel:object-types', [])
            }
        }

    def analyze_with_ai(self, question: str) -> str:
        """AI-powered OCEL analysis"""
        try:
            start=log_time("analyze_with_ai","START")
            case_events = self.events_df.sort_values('timestamp')
            events_context = []

            for _, event in case_events.iterrows():
                event_str = f"""
                Event: {event['event_id']}
                - Activity: {event['activity']}
                - Timestamp: {event['timestamp']}
                - Resource: {event['resource']}
                - Case ID: {event['case_id']}
                - Object Type: {event['object_type']}
                - Related Objects: {', '.join(event['objects'])}
                """
                events_context.append(event_str)

            context = f"""
            OCEL Log Analysis Context:
            General Statistics:
            - Total Events: {self.stats['general']['total_events']}
            - Total Cases: {self.stats['general']['total_cases']}
            - Date Range: {self.stats['general']['date_range']['start']} to {self.stats['general']['date_range']['end']}
            - Object Types: {', '.join(self.stats['object_types']['types'])}

            Detailed Event Log:
            {''.join(events_context)}

            Based on this process event log data, please answer:
            {question}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert process mining analyst. Analyze the OCEL log and provide insights."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=800
            )
            log_time("analyze_with_ai","END",start)
            return response.choices[0].message.content

        except Exception as e:
            return f"Error in AI analysis: {str(e)}"

    def create_visualizations(self):
        """Create interactive visualizations"""
        figures = {}

        # Activity Distribution
        fig_activities = px.bar(
            x=list(self.stats['activities']['distribution'].keys()),
            y=list(self.stats['activities']['distribution'].values()),
            title="Activity Distribution",
            labels={'x': 'Activity', 'y': 'Count'}
        )
        figures['activity_distribution'] = fig_activities

        # Resource Distribution
        fig_resources = px.bar(
            x=list(self.stats['resources']['distribution'].keys()),
            y=list(self.stats['resources']['distribution'].values()),
            title="Resource Distribution",
            labels={'x': 'Resource', 'y': 'Count'}
        )
        figures['resource_distribution'] = fig_resources

        return figures
