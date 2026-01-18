"""
Input validation utilities
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from loguru import logger

class InputValidator:
    """Validates various types of inputs"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email address"""
        try:
            # Use email-validator library
            validate_email(email, check_deliverability=False)
            return True, "Valid email address"
        except EmailNotValidError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Email validation error: {str(e)}")
            return False, "Email validation failed"
    
    @staticmethod
    def validate_phone(phone: str, country_code: str = "US") -> Tuple[bool, str]:
        """Validate phone number"""
        try:
            parsed_number = phonenumbers.parse(phone, country_code)
            if phonenumbers.is_valid_number(parsed_number):
                return True, "Valid phone number"
            else:
                return False, "Invalid phone number"
        except phonenumbers.NumberParseException as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Phone validation error: {str(e)}")
            return False, "Phone validation failed"
    
    @staticmethod
    def validate_date(date_str: str, date_format: str = "%Y-%m-%d") -> Tuple[bool, str]:
        """Validate date string"""
        try:
            datetime.strptime(date_str, date_format)
            return True, "Valid date"
        except ValueError as e:
            return False, f"Invalid date format. Expected: {date_format}"
        except Exception as e:
            logger.error(f"Date validation error: {str(e)}")
            return False, "Date validation failed"
    
    @staticmethod
    def validate_numeric(value: Union[str, int, float], 
                        min_val: Optional[float] = None, 
                        max_val: Optional[float] = None) -> Tuple[bool, str]:
        """Validate numeric value"""
        try:
            # Convert to float
            num = float(value)
            
            # Check bounds
            if min_val is not None and num < min_val:
                return False, f"Value must be >= {min_val}"
            
            if max_val is not None and num > max_val:
                return False, f"Value must be <= {max_val}"
            
            return True, "Valid numeric value"
            
        except (ValueError, TypeError):
            return False, "Not a valid number"
        except Exception as e:
            logger.error(f"Numeric validation error: {str(e)}")
            return False, "Numeric validation failed"
    
    @staticmethod
    def validate_string(value: str, 
                       min_length: Optional[int] = None, 
                       max_length: Optional[int] = None,
                       pattern: Optional[str] = None) -> Tuple[bool, str]:
        """Validate string value"""
        try:
            # Check length
            if min_length is not None and len(value) < min_length:
                return False, f"String must be at least {min_length} characters"
            
            if max_length is not None and len(value) > max_length:
                return False, f"String must be at most {max_length} characters"
            
            # Check pattern
            if pattern:
                if not re.match(pattern, value):
                    return False, f"String does not match pattern: {pattern}"
            
            return True, "Valid string"
            
        except Exception as e:
            logger.error(f"String validation error: {str(e)}")
            return False, "String validation failed"
    
    @staticmethod
    def validate_sql_query(query: str) -> Tuple[bool, str]:
        """Basic SQL query validation"""
        try:
            # Check for dangerous patterns
            dangerous_patterns = [
                r'(?i)drop\s+(table|database|schema)',
                r'(?i)delete\s+from',
                r'(?i)update\s+\w+\s+set',
                r'(?i)insert\s+into',
                r'(?i)alter\s+table',
                r'(?i)truncate\s+table',
                r'(?i)create\s+(table|database|schema)',
                r'(?i)grant\s+',
                r'(?i)revoke\s+',
                r';\s*--',  # SQL injection attempts
                r'union\s+select',
                r'1=1',
                r'or\s+1=1'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return False, f"Query contains potentially dangerous pattern: {pattern}"
            
            # Basic SQL syntax check
            if not query.strip().upper().startswith(('SELECT', 'WITH', 'DESC', 'SHOW', 'EXPLAIN')):
                return False, "Only SELECT, WITH, DESC, SHOW, and EXPLAIN queries are allowed"
            
            return True, "SQL query appears valid"
            
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return False, "SQL validation failed"
    
    @staticmethod
    def validate_snowflake_connection_params(params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Snowflake connection parameters"""
        required_fields = ['account', 'user', 'password', 'warehouse', 'database', 'schema']
        
        missing_fields = []
        for field in required_fields:
            if not params.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate account format
        account = params['account']
        if not re.match(r'^[a-zA-Z0-9_\-]+(\.[a-zA-Z0-9_\-]+)*$', account):
            return False, "Invalid account format"
        
        # Validate warehouse
        warehouse = params['warehouse']
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', warehouse):
            return False, "Invalid warehouse name"
        
        return True, "Connection parameters are valid"
    
    @staticmethod
    def validate_table_name(table_name: str) -> Tuple[bool, str]:
        """Validate table name"""
        try:
            # Check for SQL injection patterns
            dangerous_patterns = [
                r';',
                r'--',
                r'/\*',
                r'\*/',
                r'drop\s+',
                r'delete\s+',
                r'update\s+',
                r'insert\s+',
                r'alter\s+',
                r'truncate\s+',
                r'create\s+',
                r'grant\s+',
                r'revoke\s+'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, table_name, re.IGNORECASE):
                    return False, f"Table name contains dangerous pattern: {pattern}"
            
            # Basic table name validation
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                return False, "Invalid table name format"
            
            return True, "Valid table name"
            
        except Exception as e:
            logger.error(f"Table name validation error: {str(e)}")
            return False, "Table name validation failed"
    
    @staticmethod
    def validate_column_name(column_name: str) -> Tuple[bool, str]:
        """Validate column name"""
        try:
            # Check for SQL injection patterns
            dangerous_patterns = [
                r';',
                r'--',
                r'/\*',
                r'\*/',
                r'union\s+select',
                r'1=1',
                r'or\s+1=1'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, column_name, re.IGNORECASE):
                    return False, f"Column name contains dangerous pattern: {pattern}"
            
            # Basic column name validation
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
                return False, "Invalid column name format"
            
            return True, "Valid column name"
            
        except Exception as e:
            logger.error(f"Column name validation error: {str(e)}")
            return False, "Column name validation failed"
    
    @staticmethod
    def validate_json_schema(schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate JSON schema structure"""
        try:
            # Check required top-level fields
            required_fields = ['table_name', 'columns']
            for field in required_fields:
                if field not in schema:
                    return False, f"Missing required field: {field}"
            
            # Validate table_name
            table_name_valid, table_name_msg = InputValidator.validate_table_name(schema['table_name'])
            if not table_name_valid:
                return False, f"Invalid table_name: {table_name_msg}"
            
            # Validate columns
            if not isinstance(schema['columns'], list):
                return False, "Columns must be a list"
            
            if len(schema['columns']) == 0:
                return False, "Columns list cannot be empty"
            
            for i, column in enumerate(schema['columns']):
                if not isinstance(column, dict):
                    return False, f"Column {i} must be a dictionary"
                
                if 'name' not in column:
                    return False, f"Column {i} missing 'name' field"
                
                if 'type' not in column:
                    return False, f"Column {i} missing 'type' field"
                
                # Validate column name
                col_name_valid, col_name_msg = InputValidator.validate_column_name(column['name'])
                if not col_name_valid:
                    return False, f"Invalid column name at index {i}: {col_name_msg}"
                
                # Validate column type
                valid_types = ['VARCHAR', 'CHAR', 'STRING', 'TEXT', 'NUMBER', 'INT', 
                              'INTEGER', 'FLOAT', 'DECIMAL', 'DOUBLE', 'NUMERIC', 
                              'DATE', 'TIMESTAMP', 'DATETIME', 'TIME', 'BOOLEAN']
                
                if column['type'].upper() not in valid_types:
                    return False, f"Invalid column type at index {i}: {column['type']}"
            
            return True, "Valid JSON schema"
            
        except Exception as e:
            logger.error(f"JSON schema validation error: {str(e)}")
            return False, f"Schema validation failed: {str(e)}"