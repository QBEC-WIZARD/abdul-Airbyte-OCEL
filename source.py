import os
import json
import pandas as pd
from datetime import datetime
from airbyte_cdk.sources import Source
from airbyte_cdk.models import AirbyteMessage, AirbyteRecordMessage, Type, AirbyteStream, AirbyteCatalog
from computation.Outlier_module.ocpm_analysis import OCPMAnalyzer
from destination import LocalOCELDestination

# ✅ File expected at the root of the repo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
EVENT_LOG_PATH = os.path.join(BASE_DIR, "fx_trade_log_small.csv")


def load_event_log():
    """Load event log from a CSV file."""
    if not os.path.exists(EVENT_LOG_PATH):
        raise FileNotFoundError("❌ Event log not found. Make sure fx_trade_log_small.csv is in the root directory.")

    try:
        df = pd.read_csv(EVENT_LOG_PATH, sep=";")  
    except:
        df = pd.read_csv(EVENT_LOG_PATH)

    return df


def perform_analysis():
    """Perform OCPM analysis and return processed data."""
    df = load_event_log()
    analyzer = OCPMAnalyzer(df)
    ocel_path = analyzer.save_ocel()
    return {"ocel_path": ocel_path, "analyzer": analyzer}


class LocalEventLogSource(Source):
    def check(self, logger, config):
        """Check if log folder exists."""
        folder_path = config.get('log_folder', "")
        if not os.path.exists(folder_path):
            return {"status": "FAILED", "message": "❌ FOLDER NOT FOUND"}
        return {"status": "SUCCEEDED"}

    def discover(self, logger, config):
        """Define the schema for event logs."""
        return AirbyteCatalog(
            streams=[
                AirbyteStream(
                    name="ocel_logs",
                    json_schema={
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string"},
                            "event": {"type": "string"}
                        }
                    },
                    supported_sync_modes=["full_refresh", "incremental"]
                )
            ]
        )

    def read(self, logger, config):
        """Read event logs and send them as Airbyte messages."""
        print("📤 Sending OCEL log to Airbyte...")

        cache_OCPM = perform_analysis()['analyzer']
        ocel_data_log = cache_OCPM.ocel_data
        
        for event in ocel_data_log.get("ocel:events", []):
            yield AirbyteMessage(
                type=Type.RECORD,
                record=AirbyteRecordMessage(
                    stream="ocel_logs",
                    data=event,
                    emitted_at=int(datetime.utcnow().timestamp()) * 1000
                )
            )


if __name__ == "__main__":
    config_path = "config.json"

    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file '{config_path}' not found!")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # ✅ Initialize source and destination
    source = LocalEventLogSource()
    destination = LocalOCELDestination()

    # ✅ Run check
    print("✅ Checking source configuration...")
    check_result = source.check(None, config)
    print(check_result)

    # ✅ Discover schema
    print("\n📌 Discovering schema...")
    schema = source.discover(None, config)
    print(schema)

    # ✅ Read logs and write to destination
    print("\n📤 Reading logs and writing to destination...")
    events = []
    for message in source.read(None, config):
        if message.type == Type.RECORD:
            events.append(message.record.data)

    destination.write(events)  # Save to JSON file
    print("✅ Logs successfully saved!")
