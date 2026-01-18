"""
Helper utility functions
"""
import os
import sys
import json
import yaml
import hashlib
import random
import string
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import regex as re

def setup_project_paths():
    """Setup project paths and ensure directories exist"""
    # Project root directory
    project_root = Path(__file__).parent.parent
    
    # Create necessary directories
    directories = [
        'logs',
        'data',
        'cache',
        'exports',
        'config',
        'tmp'
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
    
    return project_root

def generate_unique_id(length: int = 12) -> str:
    """Generate a unique ID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{timestamp}_{random_str}"

def hash_string(text: str) -> str:
    """Generate SHA-256 hash of a string"""
    return hashlib.sha256(text.encode()).hexdigest()

def safe_json_loads(json_str: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"JSON decode error: {str(e)}")
        return None

def safe_yaml_load(yaml_str: str) -> Optional[Dict[str, Any]]:
    """Safely load YAML string"""
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        logger.error(f"YAML decode error: {str(e)}")
        return None

def format_bytes(size: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def format_timedelta(delta: timedelta) -> str:
    """Format timedelta to human-readable string"""
    total_seconds = int(delta.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        return f"{days}d {hours}h"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext)]
        filename = name + ext
    
    return filename

def pandas_to_dict(df: pd.DataFrame, max_rows: int = 100) -> List[Dict[str, Any]]:
    """Convert pandas DataFrame to list of dictionaries with size limit"""
    if df.empty:
        return []
    
    # Limit rows
    limited_df = df.head(max_rows)
    
    # Convert to dictionary
    records = limited_df.to_dict('records')
    
    # Convert numpy types to Python types
    for record in records:
        for key, value in record.items():
            if isinstance(value, (np.integer, np.floating)):
                record[key] = value.item()
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif pd.isna(value):
                record[key] = None
    
    return records

def dict_to_pandas(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dictionaries to pandas DataFrame"""
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def calculate_statistics(values: List[Union[int, float]]) -> Dict[str, Any]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {}
    
    values_array = np.array(values)
    
    stats = {
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q1': float(np.percentile(values_array, 25)),
        'q3': float(np.percentile(values_array, 75)),
        'iqr': float(np.percentile(values_array, 75) - np.percentile(values_array, 25))
    }
    
    return stats

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero"""
    if denominator == 0:
        return default
    return numerator / denominator

def parse_date_range(date_range: str) -> Optional[Tuple[datetime, datetime]]:
    """Parse date range string into start and end dates"""
    try:
        # Common date range patterns
        if date_range.lower() == 'today':
            start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif date_range.lower() == 'yesterday':
            start = datetime.now() - timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif date_range.lower() == 'this week':
            today = datetime.now()
            start = today - timedelta(days=today.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
        
        elif date_range.lower() == 'this month':
            today = datetime.now()
            start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = today.replace(day=28) + timedelta(days=4)
            end = next_month - timedelta(days=next_month.day)
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif date_range.lower() == 'last month':
            today = datetime.now()
            first_day_current = today.replace(day=1)
            last_day_previous = first_day_current - timedelta(days=1)
            start = last_day_previous.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = last_day_previous.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        elif ' to ' in date_range.lower() or ' - ' in date_range.lower():
            # Custom date range
            separator = ' to ' if ' to ' in date_range else ' - '
            parts = date_range.split(separator)
            
            if len(parts) == 2:
                start_str, end_str = parts
                start = datetime.strptime(start_str.strip(), '%Y-%m-%d')
                end = datetime.strptime(end_str.strip(), '%Y-%m-%d')
                end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                return None
        
        else:
            # Try to parse as single date
            start = datetime.strptime(date_range.strip(), '%Y-%m-%d')
            end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return start, end
        
    except Exception as e:
        logger.error(f"Date range parsing error: {str(e)}")
        return None

def create_progress_bar(progress: float, width: int = 40) -> str:
    """Create a text-based progress bar"""
    filled = int(width * progress)
    empty = width - filled
    bar = '█' * filled + '░' * empty
    percentage = int(progress * 100)
    return f"[{bar}] {percentage}%"

def retry_with_backoff(func: callable, max_retries: int = 3, 
                      initial_delay: float = 1.0, max_delay: float = 10.0):
    """Retry function with exponential backoff"""
    import time
    
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
        
        raise last_exception
    
    return wrapper

async def async_retry_with_backoff(func: callable, max_retries: int = 3,
                                 initial_delay: float = 1.0, max_delay: float = 10.0):
    """Retry async function with exponential backoff"""
    import asyncio
    
    async def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
        
        raise last_exception
    
    return wrapper

def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """Split items into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def validate_email_domain(email: str, allowed_domains: List[str]) -> bool:
    """Validate if email domain is in allowed list"""
    try:
        domain = email.split('@')[1]
        return domain in allowed_domains
    except (IndexError, AttributeError):
        return False

def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier to prevent injection"""
    # Remove any non-alphanumeric characters except underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
    
    # Ensure it starts with a letter or underscore
    if not sanitized or sanitized[0].isdigit():
        sanitized = '_' + sanitized
    
    return sanitized