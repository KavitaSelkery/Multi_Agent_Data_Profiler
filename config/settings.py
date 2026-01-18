"""
Configuration management with Pydantic settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings"""
    
    # # Snowflake Configuration
    # SNOWFLAKE_ACCOUNT: str = Field(default="", env="SNOWFLAKE_ACCOUNT")
    # SNOWFLAKE_USER: str = Field(default="", env="SNOWFLAKE_USER")
    # SNOWFLAKE_PASSWORD: str = Field(default="", env="SNOWFLAKE_PASSWORD")
    # SNOWFLAKE_WAREHOUSE: str = Field(default="COMPUTE_WH", env="SNOWFLAKE_WAREHOUSE")
    # SNOWFLAKE_DATABASE: str = Field(default="", env="SNOWFLAKE_DATABASE")
    # SNOWFLAKE_SCHEMA: str = Field(default="PUBLIC", env="SNOWFLAKE_SCHEMA")
    # SNOWFLAKE_ROLE: Optional[str] = Field(default=None, env="SNOWFLAKE_ROLE")
    
    # # OpenAI Configuration
    # OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    # OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    # OPENAI_TEMPERATURE: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    
    # # LangChain Configuration
    # LANGCHAIN_TRACING: bool = Field(default=False, env="LANGCHAIN_TRACING")
    # LANGCHAIN_ENDPOINT: Optional[str] = Field(default=None, env="LANGCHAIN_ENDPOINT")
    # LANGCHAIN_API_KEY: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    # LANGCHAIN_PROJECT: Optional[str] = Field(default=None, env="LANGCHAIN_PROJECT")
    
    # # Application Settings
    # SAMPLE_SIZE: int = Field(default=10000, env="SAMPLE_SIZE")
    # ANOMALY_THRESHOLD: float = Field(default=0.05, env="ANOMALY_THRESHOLD")
    # MAX_QUERY_ROWS: int = Field(default=1000, env="MAX_QUERY_ROWS")
    # sample_rate:int = Field(default=16000, env="SAMPLE_RATE")
    
    # # Agent Settings
    # AGENT_MAX_ITERATIONS: int = Field(default=10, env="AGENT_MAX_ITERATIONS")
    # AGENT_MAX_TOKENS: int = Field(default=4000, env="AGENT_MAX_TOKENS")
    # AGENT_VERBOSE: bool = Field(default=False, env="AGENT_VERBOSE")
    
    # # Logging Settings
    # LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    # LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # # Vector Store Settings
    # VECTOR_STORE_PATH: str = Field(default="./chroma_db", env="VECTOR_STORE_PATH")
    # EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    

    # Snowflake Settings
    # Pydantic-settings automatically looks for environment variables 
    # matching the field names (e.g., SNOWFLAKE_ACCOUNT)
    SNOWFLAKE_ACCOUNT: str = ""
    SNOWFLAKE_USER: str = ""
    SNOWFLAKE_PASSWORD: str = ""
    SNOWFLAKE_WAREHOUSE: str = "COMPUTE_WH"
    SNOWFLAKE_DATABASE: str = ""
    SNOWFLAKE_SCHEMA: str = "PUBLIC"
    SNOWFLAKE_ROLE: Optional[str] = None
    
    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.1
    
    # LangChain Settings
    LANGCHAIN_TRACING: bool = False
    LANGCHAIN_ENDPOINT: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None
    
    # Agent Settings
    SAMPLE_SIZE: int = 10000
    ANOMALY_THRESHOLD: float = 0.05
    MAX_QUERY_ROWS: int = 1000
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_MAX_TOKENS: int = 2000
    sample_rate:int = 16000
    AGENT_VERBOSE: bool = False


    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Vector Store Settings
    VECTOR_STORE_PATH: Optional[str] = str(PROJECT_ROOT / "chromadb_data")
    EMBEDDING_MODEL: Optional[str] = "text-embedding-3-small"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignores extra environment variables
    )

    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"
    
    def validate_config(self):
        """Validate required configuration"""
        missing = []
        
        if not self.SNOWFLAKE_ACCOUNT:
            missing.append('SNOWFLAKE_ACCOUNT')
        if not self.SNOWFLAKE_USER:
            missing.append('SNOWFLAKE_USER')
        if not self.SNOWFLAKE_PASSWORD:
            missing.append('SNOWFLAKE_PASSWORD')
        if not self.SNOWFLAKE_DATABASE:
            missing.append('SNOWFLAKE_DATABASE')
        if not self.OPENAI_API_KEY:
            missing.append('OPENAI_API_KEY')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True