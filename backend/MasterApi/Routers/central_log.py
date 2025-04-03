import time
import logging
import os
from functools import wraps

# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging with explicit file handler
log_file = os.path.join(log_dir, "execution_times.log")

# Create logger
logger = logging.getLogger("execution_timer")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(file_handler)

# Function to log start and end times
def log_time(func_name, event, start_time=None):
    timestamp = time.time()

    if event.lower() == "end" and start_time is not None:
        duration = timestamp - start_time
        logger.info(f"{func_name} | {event}: {timestamp} | Duration: {duration:.6f} seconds")
        print(f"Logging to {os.path.abspath(log_file)}: {func_name} | {event}: {timestamp} | Duration: {duration:.6f} seconds")
    else:
        logger.info(f"{func_name} | {event}: {timestamp}")
        print(f"Logging to {os.path.abspath(log_file)}: {func_name} | {event}: {timestamp}")
    
    return timestamp

# Decorator for automatically timing functions
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = log_time(func_name, "start")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            log_time(func_name, "end", start_time)
            
    return wrapper