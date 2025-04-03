from openai import AzureOpenAI

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")  # Get key from environment variable

def get_azure_openai_client():
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-01",
        azure_endpoint="https://smartcall.openai.azure.com/"
    )
    return client
