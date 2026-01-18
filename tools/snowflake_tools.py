"""
Snowflake connection and query tools
"""
import snowflake.connector
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from loguru import logger
from contextlib import contextmanager
import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings

class SnowflakeManager:
    """Manager for Snowflake connections and operations"""
    
    def __init__(self):
        """Initialize Snowflake manager"""
        self.settings = Settings()
        self.connection = None
        self.is_connected = False
    
    def connect(self, account: str, user: str, password: str, 
                warehouse: str, database: str, schema: str, 
                role: Optional[str] = None) -> Tuple[bool, str]:
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role=role,
                client_session_keep_alive=True
            )
            
            self.is_connected = True
            logger.info(f"Connected to Snowflake: {account}/{database}/{schema}")
            return True, "Connected successfully!"
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False, f"Connection failed: {str(e)}"
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            yield cursor
        except Exception as e:
            logger.error(f"Cursor error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query)
                
                # Get column names
                columns = [col[0] for col in cursor.description] if cursor.description else []
                
                # Fetch results
                data = cursor.fetchall()
                
                # Create DataFrame
                if columns and data:
                    df = pd.DataFrame(data, columns=columns)
                elif data:
                    df = pd.DataFrame(data)
                    df.columns = [f'Column_{i}' for i in range(len(df.columns))]
                else:
                    df = pd.DataFrame()
                
                return df
                
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            error_df = pd.DataFrame({
                'Error': [str(e)],
                'Query': [query[:200]]
            })
            return error_df
    
    def get_tables(self) -> List[str]:
        """Get list of tables in current schema"""
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = CURRENT_SCHEMA()
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
            
            df = self.execute_query(query)
            return df['TABLE_NAME'].tolist() if not df.empty else []
            
        except Exception as e:
            logger.error(f"Get tables error: {str(e)}")
            return []
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get detailed column information for a specific table"""
        try:
            query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                ordinal_position,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns 
            WHERE table_name = '{table_name.upper()}'
            AND table_schema = CURRENT_SCHEMA()
            ORDER BY ordinal_position
            """
            
            return self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Get schema error: {str(e)}")
            return pd.DataFrame()
    
    def get_sample_data(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """Get sample data from table"""
        try:
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            return self.execute_query(query)
            
        except Exception as e:
            logger.error(f"Get sample data error: {str(e)}")
            return pd.DataFrame()
    
    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            query = f'SELECT COUNT(*) as row_count FROM "{table_name}"'
            df = self.execute_query(query)
            return df['ROW_COUNT'].iloc[0] if not df.empty else 0
            
        except Exception as e:
            logger.error(f"Get row count error: {str(e)}")
            return 0
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if a SQL query is valid"""
        try:
            explain_query = f"EXPLAIN {query}"
            self.execute_query(explain_query)
            return True, "Query is valid"
            
        except Exception as e:
            error_msg = str(e)
            if "snowflake.connector.errors" in error_msg:
                error_parts = error_msg.split(": ")
                if len(error_parts) > 1:
                    error_msg = error_parts[-1]
            return False, error_msg
    
    def close(self):
        """Close the connection"""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info("Snowflake connection closed")