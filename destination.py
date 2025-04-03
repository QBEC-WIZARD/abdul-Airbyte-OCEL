import json
import os
from airbyte_cdk.destinations import Destination


class LocalOCELDestination(Destination):
    def __init__(self, file_path="final_output.json"):
        self.file_path = file_path

    def write(self, records):
        """Writes the received records into a JSON file."""
        if not records:
            print("⚠ No records to save.")
            return

        existing_data = []
        
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []

        # Append new records
        existing_data.extend(records)

        # Save back to file
        with open(self.file_path, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"✅ {len(records)} logs saved to {self.file_path}")

    def check(self, logger, config):
        """Checks if the destination is valid."""
        return {"status": "SUCCEEDED"}

    def discover(self, logger, config):
        """Returns schema information (not required for local storage)."""
        return {}
