from openai import AzureOpenAI
import os
import json
from collections import defaultdict
import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils import get_azure_openai_client
import requests
import datetime
import os

@dataclass
class ObjectType:
    """Represents an object type in the OCPM model"""
    name: str
    activities: List[str]
    attributes: List[str]
    relationships: List[str]


@dataclass
class LogSummary:
    """Stores the summary statistics of the event log"""
    unique_activities: List[str]
    case_resources: Dict[str, List[str]]
    activity_attributes: Dict[str, List[str]]
    case_activities: Dict[str, List[str]]
    column_values: Dict[str, List[str]]


def analyze_log_file(df: pd.DataFrame) -> LogSummary:
    """
    Analyzes the entire log file to extract unique values and relationships.
    Handles standard columns (case_id, activity, timestamp, resource) separately
    from case-specific attribute columns.
    """
    # Standard columns that should be present in all event logs
    STANDARD_COLUMNS = ['case_id', 'activity', 'timestamp', 'resource']

    # Verify all standard columns are present
    missing_columns = [col for col in STANDARD_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing standard columns: {missing_columns}")

    # Get all unique activities
    unique_activities = df['activity'].unique().tolist()

    # Get resources used per case
    case_resources = defaultdict(set)
    for _, row in df.iterrows():
        case_resources[row['case_id']].add(row['resource'])
    case_resources = {k: list(v) for k, v in case_resources.items()}

    # Get attributes associated with each activity
    # Only consider non-standard columns as potential attributes
    activity_attributes = defaultdict(set)
    attribute_columns = [col for col in df.columns if col not in STANDARD_COLUMNS]

    for _, row in df.iterrows():
        for col in attribute_columns:
            if pd.notna(row[col]):
                activity_attributes[row['activity']].add(col)
    activity_attributes = {k: list(v) for k, v in activity_attributes.items()}

    # Get activities per case
    case_activities = defaultdict(list)
    for _, row in df.iterrows():
        case_activities[row['case_id']].append(row['activity'])

    # Get unique values for each non-timestamp column
    column_values = {}
    # Add standard columns first (except timestamp)
    for col in STANDARD_COLUMNS:
        if col != 'timestamp':
            column_values[col] = df[col].unique().tolist()

    # Add case-specific attribute columns
    for col in attribute_columns:
        column_values[col] = df[col].unique().tolist()

    return LogSummary(
        unique_activities=unique_activities,
        case_resources=case_resources,
        activity_attributes=activity_attributes,
        case_activities=dict(case_activities),
        column_values=column_values
    )


def read_industry_context(context_file: str) -> str:
    """
    Reads industry context from the specified file

    Args:
        context_file: Path to the context file

    Returns:
        String containing the industry context
    """
    try:
        with open(context_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Industry context file {context_file} not found")
        return ""


def chunk_dataframe(df, chunk_size=1000):
    """Split dataframe into chunks of specified size"""
    num_chunks = math.ceil(len(df) / chunk_size)
    return [df[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def merge_object_types(object_types_list):
    """
    Merge object types from multiple chunks while preserving activity order.
    Returns a dictionary structure that can be easily serialized to JSON.
    """
    merged = {}

    # Track the original order of activities as they first appear
    activity_order = {}

    # First pass: collect all activities and their first appearance order
    for obj_types in object_types_list:
        for obj_name, obj_data in obj_types.items():
            if obj_name not in activity_order:
                activity_order[obj_name] = []

            # Add new activities while preserving their order of appearance
            for activity in obj_data.get('activities', []):
                if activity not in activity_order[obj_name]:
                    activity_order[obj_name].append(activity)

    # Second pass: merge the data while maintaining order
    for obj_types in object_types_list:
        for obj_name, obj_data in obj_types.items():
            # Ensure all required fields are present
            required_fields = ['activities', 'attributes', 'relationships']
            if not all(field in obj_data for field in required_fields):
                print(f"Warning: Missing required fields in object type {obj_name}")
                continue

            if obj_name not in merged:
                merged[obj_name] = {
                    'name': obj_name,
                    'activities': set(),
                    'attributes': set(obj_data['attributes']),
                    'relationships': set(obj_data['relationships'])
                }
            else:
                # Update sets for attributes and relationships
                merged[obj_name]['attributes'].update(obj_data['attributes'])
                merged[obj_name]['relationships'].update(obj_data['relationships'])

            # Update activities while preserving order
            merged[obj_name]['activities'].update(obj_data['activities'])

    # Final pass: order activities according to their first appearance
    for obj_name in merged:
        # Use the tracked order to sort activities
        ordered_activities = [act for act in activity_order[obj_name]
                              if act in merged[obj_name]['activities']]

        # Add any activities that might have been missed (shouldn't happen, but safe)
        remaining_activities = merged[obj_name]['activities'] - set(ordered_activities)
        ordered_activities.extend(sorted(remaining_activities))  # Only sort remaining if any

        # Update with ordered activities list
        merged[obj_name]['activities'] = ordered_activities

        # Still sort attributes and relationships (order doesn't matter for these)
        merged[obj_name]['attributes'] = sorted(list(merged[obj_name]['attributes']))
        merged[obj_name]['relationships'] = sorted(list(merged[obj_name]['relationships']))

    return merged


def save_enhanced_prompt(prompt: str):
    """
    Saves the enhanced prompt to a file with timestamp in the filename.
    Creates the output directory if it doesn't exist.

    Args:
        prompt: The enhanced prompt string to save
    """

    # Create output directory if it doesn't exist
    # os.makedirs('ocpm_output', exist_ok=True)

    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create output directories if they don't exist
    # os.makedirs('ocpm_output', exist_ok=True)

    # filename = f'ocpm_output/enhanced_prompt_{timestamp}.txt'
    api_response = f'api_response/enhanced_prompt_{timestamp}.txt'

    # # Write the prompt to file
    # with open(filename, 'w') as f:
    #     f.write(prompt)
    # print(f"Enhanced prompt saved to {filename}")

    # Write the prompt to api_response
    os.makedirs('api_response', exist_ok=True)
    with open(api_response, 'w') as f:
        f.write(prompt)
    print(f"Enhanced prompt also saved to {api_response}")

    # Send the prompt to an API endpoint
    # api_endpoint = "http://127.0.0.1:8000/enhanced_prompt"
    # try:
    #     response = requests.post(api_endpoint, json={"prompt": prompt})
    #     response.raise_for_status()
    #     print(f"Prompt successfully sent to {api_endpoint}")
    # except requests.exceptions.RequestException as e:
    #     print(f"Failed to send prompt to API: {e}")

    ## Sending API to send the prompt

    


def create_enhanced_prompt(log_summary: LogSummary, industry_context: str) -> str:
    """
    Creates an enhanced prompt for converting traditional event logs to OCEL format.

    Args:
        log_summary: Summary statistics of the event log
        industry_context: Domain-specific context for better object identification

    Returns:
        A detailed prompt string for AI model to extract OCEL objects
    """
    activities_str = json.dumps(log_summary.unique_activities, indent=2)
    case_resources_str = json.dumps(dict(list(log_summary.case_resources.items())[:5]), indent=2)
    activity_attrs_str = json.dumps(log_summary.activity_attributes, indent=2)

    # Example JSON structure for reference
    example_json = '''
{
    "Trade": {
        "activities": ["Create Trade", "Execute Trade", "Modify Trade"],
        "attributes": ["trade_id", "currency_pair", "amount", "trade_type", "status"],
        "relationships": ["Order", "Position", "Client"]
    },
    "Order": {
        "activities": ["Place Order", "Cancel Order", "Modify Order"],
        "attributes": ["order_id", "order_type", "quantity", "price", "status"],
        "relationships": ["Trade", "Client", "Account"]
    }
}'''

    return f"""Based on the provided event log analysis and industry context, identify object types and their relationships to activities/events for conversion into the Object-Centric Event Log (OCEL) format.

Industry Context:
{industry_context}

Log Analysis Details:
- Unique Activities: {activities_str}
- Sample Case-Resource Mappings: {case_resources_str}
- Activity Attributes: {activity_attrs_str}
- Number of Unique Cases: {len(log_summary.case_activities)}
- Available Columns: {list(log_summary.column_values.keys())}

Object Identification Guidelines:
1. **Analyze Activities to Identify Potential Business Objects:**
   - Focus on nouns within activity descriptions that represent business entities.
   - Consider entities that persist across multiple activities and have distinct lifecycles.
   - Identify objects that are created, modified, or referenced by activities.

2. **Activity-Object Association Rules:**
   - Associate each activity with relevant object types based on its context.
   - Activities such as "Create X" or "Update X" typically indicate that 'X' is an object.
   - Ensure that each activity is linked to at least one object type.

3. **Attribute Identification Guidelines:**
   - Map relevant columns from the event log to object attributes.
   - Convert case attributes to object attributes where appropriate.
   - Ensure proper mapping of timestamps and resources.
   - Include unique identifiers (e.g., IDs) as attributes for corresponding objects.

4. **Relationship Identification Guidelines:**
   - Determine how objects interact during activities.
   - Identify parent-child or hierarchical relationships between objects.
   - Look for objects that share common activities or are frequently associated.
   - Consider relationships implied by foreign key references or data linkages.
   
5. **Activity Sequencing Rules:**
   - Identify mandatory start activities (e.g., "Trade Initiated" for trades)
   - Identify mandatory end activities (e.g., "Final Settlement" for trades)
   - Maintain business process order within each object type
   - Consider dependencies between activities
   - Respect regulatory and operational sequences

6. **Sequence Validation Rules:**
   - Start activities must appear first in the sequence
   - End activities must appear last in the sequence
   - Intermediate activities must follow business logic order
   - Validation activities must precede execution activities
   - Settlement activities must follow execution activities

7. **Cross-Object Sequence Dependencies:**
   - Order activities must precede related Trade activities
   - Market Data validation must precede Trade execution
   - Position updates must follow Trade execution
   - Client checks must precede Trade execution

Output Requirements:
1. **JSON Structure:**
   - Each key represents an object type name.
   - Each object type contains the following fields:
     - "activities": List of activities that create, modify, or reference this object.
     - "attributes": List of data fields associated with this object.
     - "relationships": List of other object types this object interacts with.

2. **Naming Conventions:**
   - Use PascalCase for object type names (e.g., "Trade", "Order").
   - Retain original activity names from the log.
   - Use lowercase with underscores for attribute names.
   - Ensure consistency in relationship naming aligned with object type names.

Example Output Format:
{example_json}

Additional Instructions:
- Ensure comprehensive mapping of all activities to relevant object types.
- Include all pertinent attributes from the available columns.
- Define clear and meaningful relationships between objects.
- Validate that object types form a connected graph through their relationships.
- Incorporate industry-specific context when defining object types and relationships.

Your response must:
1. Adhere to the exact JSON format provided in the example.
2. Include all identified object types with their complete activities, attributes, and relationships.
3. Maintain consistency in naming conventions and structure.
4. Cover all activities and relevant attributes from the event log comprehensively.

**Process Flow Analysis Rules:**
1. **Activity Order Detection:**
   - Analyze case timestamps to determine natural progression
   - Look for temporal patterns across multiple cases
   - Identify consistent activity sequences in the event log
   - Consider activity names that imply order (e.g., "Initiated", "Completed", "Final")

2. **Logical Flow Dependencies:**
   - Creation/Initiation activities must come first
   - Validation/Check activities must precede their target activities
   - Core processing activities follow validation
   - Assessment/Review activities follow processing
   - Settlement/Completion activities come last
   - Final/End activities must be at the end

3. **Timeline-Based Sequencing:**
   - Use timestamp analysis to validate activity order
   - Consider minimum time gaps between activities
   - Look for parallel vs. sequential patterns
   - Identify activities that consistently occur early/late in cases
   - Detect standard waiting periods between activities

4. **Cross-Object Dependencies:**
   - Identify activities that must complete in one object before another starts
   - Consider input/output relationships between objects
   - Maintain consistency in related object sequences
   - Ensure dependent activities across objects maintain proper order

5. **Validation Rules:**
   - Every object type must have clear start and end activities
   - No validation activity should follow its target execution
   - Review activities cannot precede their subject activities
   - Settlement activities must come after core processing
   - Maintain logical business process flow

"""


def process_chunk(chunk: pd.DataFrame, enhanced_prompt: str, client: AzureOpenAI, chunk_num: int) -> Optional[dict]:
    """Process a single chunk and return validated results"""
    try:
        log_text = chunk.to_json(orient='records')
        full_prompt = f"{enhanced_prompt}\n\nChunk Data:\n{log_text}"

        print(f"\nProcessing chunk {chunk_num}:")
        print(f"Chunk size: {len(chunk)} rows")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a process mining expert. Return only valid JSON matching the exact format specified in the prompt."
                    },
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )

            # Log raw response for debugging
            raw_response = response.choices[0].message.content
            print(f"\nChunk {chunk_num} Raw Response:")
            print("Response length:", len(raw_response))
            print("First 500 chars of response:")
            print(raw_response[:500])
            print("\nLast 500 chars of response:")
            print(raw_response[-500:] if len(raw_response) > 500 else "")

            try:
                # Parse and validate response
                chunk_result = json.loads(raw_response)
                print(f"Successfully parsed JSON for chunk {chunk_num}")

                # Basic structure validation
                if not isinstance(chunk_result, dict):
                    print(f"Error: Response is not a dictionary. Type: {type(chunk_result)}")
                    return None

                # Validate each object type
                for obj_name, obj_data in chunk_result.items():
                    print(f"Validating object type: {obj_name}")
                    if not isinstance(obj_data, dict):
                        print(f"Error: Object {obj_name} is not a dictionary")
                        return None

                    required_fields = ['activities', 'attributes', 'relationships']
                    for field in required_fields:
                        if field not in obj_data:
                            print(f"Error: Missing required field '{field}' in object {obj_name}")
                            return None
                        if not isinstance(obj_data[field], list):
                            print(f"Error: Field '{field}' in object {obj_name} is not a list")
                            return None
                        if not all(isinstance(item, str) for item in obj_data[field]):
                            print(f"Error: Field '{field}' in object {obj_name} contains non-string items")
                            return None

                print(f"Successfully validated chunk {chunk_num} with {len(chunk_result)} object types")
                return chunk_result

            except json.JSONDecodeError as e:
                print(f"JSON parsing error in chunk {chunk_num}:")
                print(f"Error details: {str(e)}")
                print("Error location in response:", e.pos)
                print("Context around error:")
                start = max(0, e.pos - 50)
                end = min(len(raw_response), e.pos + 50)
                print(raw_response[start:end])
                return None

        except Exception as api_error:
            print(f"Azure API error in chunk {chunk_num}:")
            print(f"Error type: {type(api_error)}")
            print(f"Error details: {str(api_error)}")
            return None

    except Exception as e:
        print(f"Unexpected error processing chunk {chunk_num}:")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        return None



def process_log_file(df: pd.DataFrame, chunk_size=1000):
    """Process the log file with enhanced validation and error handling"""
    try:
        print("Analyzing log file...")
        log_summary = analyze_log_file(df)

        # Read industry context
        context_file = 'staging/industry_context/fx_trade_log_context.txt'
        industry_context = read_industry_context(context_file)

        # Create enhanced prompt
        enhanced_prompt = create_enhanced_prompt(log_summary, industry_context)

        save_enhanced_prompt(enhanced_prompt)

        # Initialize client
        client = get_azure_openai_client()

        # Process chunks
        chunks = chunk_dataframe(df, chunk_size)
        all_object_types = []

        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}")
            result = process_chunk(chunk, enhanced_prompt, client, i)
            if result:
                all_object_types.append(result)

        # Check if we have any valid results
        if not all_object_types:
            print("Error: No valid chunks were processed")
            return {}

        # Merge and save results
        merged_json = merge_object_types(all_object_types)

        if not merged_json:
            print("Error: Merge operation resulted in empty object types")
            return {}

        # Save the merged results directly - they're already in the correct format
        output_path = 'api_response/output_ocel.json'
        os.makedirs('api_response', exist_ok=True)

        api_response="api_response/output_ocel.json"

        with open(output_path, 'w') as f:
            json.dump(merged_json, f, indent=2)

        with open(api_response, 'w') as f:
            json.dump(merged_json, f, indent=2)

        # Send the file to an API endpoint

        # api_endpoint = "http://127.0.0.1:8000/output_ocel"

        # try:
        #     response = requests.post(api_endpoint, json=merged_json)
        #     response.raise_for_status()
        #     print(f"File successfully uploaded to {api_endpoint}")
        # except requests.exceptions.RequestException as e:
        #     print(f"Failed to upload file: {e}")

        print(f"\nSuccessfully saved {len(merged_json)} object types to {output_path}")

        # Only convert to ObjectType instances after saving JSON
        object_types = {}
        for obj_name, obj_data in merged_json.items():
            object_types[obj_name] = ObjectType(
                name=obj_name,
                activities=obj_data['activities'],
                attributes=obj_data['attributes'],
                relationships=obj_data['relationships']
            )

        return object_types

    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        raise