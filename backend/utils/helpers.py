import pandas as pd

def extract_json_schema(data, parent_key=""):
    schema = {}

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            schema[full_key] = type(value).__name__  # Get type as string

            # Recursively check nested dictionaries or lists
            if isinstance(value, (dict, list)):
                schema.update(extract_json_schema(value, full_key))

    elif isinstance(data, list) and data:
        schema[parent_key] = f"List[{type(data[0]).__name__}]"  # Get type of first element
        if isinstance(data[0], (dict, list)):
            schema.update(extract_json_schema(data[0], parent_key))

    return schema

def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')



