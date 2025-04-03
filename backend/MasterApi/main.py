# import the required modules
import os
# Importing the fastapi for the API
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# import logging
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from tornado.websocket import WebSocketClosedError  # Correct import
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the directory when the server is initiated
if not os.path.exists("api_response"):
    os.makedirs("api_response")
    logger.info('Directory created successfully')


#Importing all the static files and CORS middleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Importing the FastAPI


from fastapi import File, UploadFile

from fastapi import FastAPI
from backend.MasterApi.routes import router
from backend.MasterApi.Routers.Process_discovery_router import pd_router
from backend.MasterApi.Routers.Outlier_module_router import out_router
from backend.MasterApi.Routers.gap_analysis_module import gap_router
from backend.MasterApi.Routers.FMEA_analysis_module import fmea_router


app = FastAPI()
app.include_router(router)
app.include_router(pd_router)
app.include_router(out_router)
app.include_router(gap_router)
app.include_router(fmea_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/page1")
async def page1(request: Request):
    return templates.TemplateResponse("page1.html", {"request": request})

#make an api end point for uploading event log



## Funtion to ensure that the folder is cleared before the server starts
# def clear_folder(folder_path: str):
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 os.rmdir(file_path)
#         except Exception as e:
#             logger.error(f'Failed to delete {file_path}. Reason: {e}')

# @app.on_event("startup")
# async def startup_event():
#     folder_path = "api_response"
#     clear_folder(folder_path)
#     logger.info('Folder contents cleared successfully')

