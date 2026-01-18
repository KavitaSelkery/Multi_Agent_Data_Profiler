"""
Anomaly detection agent using advanced techniques
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from loguru import logger
from langchain_core.tools import Tool
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeManager
from tools.data_tools import DataAnalyzer
from config.prompts import ANOMALY_AGENT_PROMPT

class AnomalyAgent(BaseAgent):
    """Agent for anomaly detection tasks"""
    
    def __init__(self, snowflake_manager: SnowflakeManager):
        """Initialize anomaly detection agent"""
        self.snowflake_manager = snowflake_manager
        self.data_analyzer = DataAnalyzer()
        
        # Define agent tools
        tools = [
            self.detect_numeric_anomalies_tool(),
            self.detect_temporal_anomalies_tool(),
            self.detect_categorical_anomalies_tool(),
            self.detect_pattern_anomalies_tool(),
            self.analyze_anomaly_causes_tool()
        ]
        
        super().__init__(
            name="Anomaly Detection Agent",
            description="Detects anomalies, outliers, and unusual patterns in data",
            tools=tools
        )
    
    def detect_numeric_anomalies_tool(self) -> Tool:
        """Tool to detect numeric anomalies"""
        
        def detect_numeric_anomalies(table_name: str, column_name: str = None, method: str = "auto") -> str:
            """Detect anomalies in numeric columns using various methods"""
            try:
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Filter to specific column if provided
                if column_name and column_name in sample_df.columns:
                    columns_to_check = [column_name]
                else:
                    # Get all numeric columns
                    columns_to_check = [
                        col for col in sample_df.columns 
                        if pd.api.types.is_numeric_dtype(sample_df[col])
                    ]
                
                if not columns_to_check:
                    return f"No numeric columns found in table: {table_name}"
                
                # Detect anomalies
                all_anomalies = []
                for col in columns_to_check:
                    anomalies = self.data_analyzer.detect_anomalies(
                        sample_df[[col]], 
                        method=method
                    )
                    all_anomalies.extend(anomalies)
                
                # Format results
                if not all_anomalies:
                    return "No numeric anomalies detected"
                
                results = []
                for anomaly in all_anomalies:
                    results.append(
                        f"Column: {anomaly['column']}, "
                        f"Type: {anomaly['anomaly_type']}, "
                        f"Count: {anomaly['count']} ({anomaly['percentage']:.1f}%), "
                        f"Severity: {anomaly['severity']}"
                    )
                
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"Numeric anomaly detection error: {str(e)}")
                return f"Error detecting numeric anomalies: {str(e)}"
        
        return Tool(
            name="detect_numeric_anomalies",
            func=detect_numeric_anomalies,
            description="Detect anomalies in numeric columns using IQR, Z-Score, or Isolation Forest methods"
        )
    
    def detect_temporal_anomalies_tool(self) -> Tool:
        """Tool to detect temporal anomalies"""
        
        def detect_temporal_anomalies(table_name: str, date_column: str) -> str:
            """Detect anomalies in date/time columns"""
            try:
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                if date_column not in sample_df.columns:
                    return f"Column {date_column} not found in table"
                
                # Convert to datetime
                date_series = pd.to_datetime(sample_df[date_column], errors='coerce')
                
                if date_series.isnull().all():
                    return f"Column {date_column} does not contain valid dates"
                
                clean_series = date_series.dropna()
                
                # Detect anomalies
                anomalies = []
                
                # Future dates
                now = pd.Timestamp.now()
                future_dates = clean_series[clean_series > now]
                if not future_dates.empty:
                    anomalies.append({
                        'type': 'future_dates',
                        'count': len(future_dates),
                        'max_date': future_dates.max().strftime('%Y-%m-%d')
                    })
                
                # Very old dates
                old_threshold = pd.Timestamp('1900-01-01')
                old_dates = clean_series[clean_series < old_threshold]
                if not old_dates.empty:
                    anomalies.append({
                        'type': 'very_old_dates',
                        'count': len(old_dates),
                        'min_date': old_dates.min().strftime('%Y-%m-%d')
                    })
                
                # Missing dates (gaps)
                if len(clean_series) > 10:
                    sorted_dates = clean_series.sort_values()
                    date_diffs = sorted_dates.diff().dt.days
                    large_gaps = date_diffs[date_diffs > 365]  # Gaps > 1 year
                    if not large_gaps.empty:
                        anomalies.append({
                            'type': 'large_date_gaps',
                            'count': len(large_gaps),
                            'max_gap_days': int(large_gaps.max())
                        })
                
                if not anomalies:
                    return "No temporal anomalies detected"
                
                # Format results
                results = []
                for anomaly in anomalies:
                    result_str = f"Type: {anomaly['type']}, Count: {anomaly['count']}"
                    if 'max_date' in anomaly:
                        result_str += f", Max Date: {anomaly['max_date']}"
                    if 'min_date' in anomaly:
                        result_str += f", Min Date: {anomaly['min_date']}"
                    if 'max_gap_days' in anomaly:
                        result_str += f", Max Gap: {anomaly['max_gap_days']} days"
                    results.append(result_str)
                
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"Temporal anomaly detection error: {str(e)}")
                return f"Error detecting temporal anomalies: {str(e)}"
        
        return Tool(
            name="detect_temporal_anomalies",
            func=detect_temporal_anomalies,
            description="Detect anomalies in date/time columns (future dates, old dates, gaps)"
        )
    
    def detect_categorical_anomalies_tool(self) -> Tool:
        """Tool to detect categorical anomalies"""
        
        def detect_categorical_anomalies(table_name: str, column_name: str) -> str:
            """Detect anomalies in categorical/text columns"""
            try:
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                if column_name not in sample_df.columns:
                    return f"Column {column_name} not found in table"
                
                series = sample_df[column_name].astype(str)
                
                # Detect anomalies
                anomalies = []
                
                # Very long strings
                length_series = series.str.len()
                long_strings = series[length_series > 1000]
                if not long_strings.empty:
                    anomalies.append({
                        'type': 'very_long_strings',
                        'count': len(long_strings),
                        'max_length': int(length_series.max())
                    })
                
                # Inconsistent casing
                if len(series) > 10:
                    upper_count = series.str.isupper().sum()
                    lower_count = series.str.islower().sum()
                    mixed_count = len(series) - upper_count - lower_count
                    
                    if upper_count > 0 and lower_count > 0 and mixed_count > 0:
                        anomalies.append({
                            'type': 'inconsistent_casing',
                            'upper_count': int(upper_count),
                            'lower_count': int(lower_count),
                            'mixed_count': int(mixed_count)
                        })
                
                # Special characters
                special_char_pattern = r'[^\w\s\-\.\,]'
                special_char_count = series.str.contains(special_char_pattern, regex=True).sum()
                if special_char_count > len(series) * 0.3:
                    anomalies.append({
                        'type': 'high_special_characters',
                        'count': int(special_char_count),
                        'percentage': (special_char_count / len(series)) * 100
                    })
                
                # Dominant values
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    most_common_percentage = (value_counts.iloc[0] / len(series)) * 100
                    if most_common_percentage > 90:
                        anomalies.append({
                            'type': 'dominant_value',
                            'value': value_counts.index[0],
                            'percentage': most_common_percentage
                        })
                
                if not anomalies:
                    return "No categorical anomalies detected"
                
                # Format results
                results = []
                for anomaly in anomalies:
                    result_str = f"Type: {anomaly['type']}"
                    if 'count' in anomaly:
                        result_str += f", Count: {anomaly['count']}"
                    if 'max_length' in anomaly:
                        result_str += f", Max Length: {anomaly['max_length']}"
                    if 'percentage' in anomaly:
                        result_str += f", Percentage: {anomaly['percentage']:.1f}%"
                    if 'value' in anomaly:
                        result_str += f", Value: '{anomaly['value']}'"
                    results.append(result_str)
                
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"Categorical anomaly detection error: {str(e)}")
                return f"Error detecting categorical anomalies: {str(e)}"
        
        return Tool(
            name="detect_categorical_anomalies",
            func=detect_categorical_anomalies,
            description="Detect anomalies in categorical/text columns (long strings, inconsistent casing, special chars)"
        )
    
    def detect_pattern_anomalies_tool(self) -> Tool:
        """Tool to detect pattern anomalies"""
        
        def detect_pattern_anomalies(table_name: str) -> str:
            """Detect pattern-based anomalies across the entire table"""
            try:
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Detect anomalies
                anomalies = []
                
                # Duplicate rows
                duplicate_count = sample_df.duplicated().sum()
                if duplicate_count > 0:
                    anomalies.append({
                        'type': 'duplicate_rows',
                        'count': int(duplicate_count),
                        'percentage': (duplicate_count / len(sample_df)) * 100
                    })
                
                # High null percentage columns
                null_percentages = sample_df.isnull().mean() * 100
                high_null_columns = null_percentages[null_percentages > 30]
                for column, null_pct in high_null_columns.items():
                    anomalies.append({
                        'type': 'high_null_percentage',
                        'column': column,
                        'percentage': float(null_pct)
                    })
                
                # Constant columns
                for column in sample_df.columns:
                    if sample_df[column].nunique() == 1 and len(sample_df) > 10:
                        anomalies.append({
                            'type': 'constant_column',
                            'column': column,
                            'value': str(sample_df[column].iloc[0])
                        })
                
                if not anomalies:
                    return "No pattern anomalies detected"
                
                # Format results
                results = []
                for anomaly in anomalies:
                    result_str = f"Type: {anomaly['type']}"
                    if 'count' in anomaly:
                        result_str += f", Count: {anomaly['count']}"
                    if 'percentage' in anomaly:
                        result_str += f", Percentage: {anomaly['percentage']:.1f}%"
                    if 'column' in anomaly:
                        result_str += f", Column: {anomaly['column']}"
                    if 'value' in anomaly:
                        result_str += f", Value: {anomaly['value']}"
                    results.append(result_str)
                
                return "\n".join(results)
                
            except Exception as e:
                logger.error(f"Pattern anomaly detection error: {str(e)}")
                return f"Error detecting pattern anomalies: {str(e)}"
        
        return Tool(
            name="detect_pattern_anomalies",
            func=detect_pattern_anomalies,
            description="Detect pattern-based anomalies (duplicate rows, high null percentages, constant columns)"
        )
    
    def analyze_anomaly_causes_tool(self) -> Tool:
        """Tool to analyze root causes of anomalies"""
        
        def analyze_anomaly_causes(table_name: str, anomaly_description: str) -> str:
            """Analyze potential root causes for detected anomalies"""
            try:
                # This is a more complex analysis that might involve multiple queries
                # For now, provide intelligent suggestions based on anomaly type
                
                anomaly_lower = anomaly_description.lower()
                
                suggestions = []
                
                if 'future' in anomaly_lower or 'date' in anomaly_lower:
                    suggestions.append("• Check for incorrect system clocks or timezone settings")
                    suggestions.append("• Verify date entry processes and validation rules")
                    suggestions.append("• Review ETL processes for date transformations")
                
                if 'null' in anomaly_lower:
                    suggestions.append("• Review data entry validation rules")
                    suggestions.append("• Check ETL transformation logic for missing value handling")
                    suggestions.append("• Verify data source completeness")
                
                if 'duplicate' in anomaly_lower:
                    suggestions.append("• Review unique constraint enforcement")
                    suggestions.append("• Check data import processes for duplicate prevention")
                    suggestions.append("• Verify business logic for duplicate detection")
                
                if 'outlier' in anomaly_lower or 'extreme' in anomaly_lower:
                    suggestions.append("• Verify data entry validation ranges")
                    suggestions.append("• Check for data transformation errors")
                    suggestions.append("• Review business rules for acceptable value ranges")
                
                if 'type' in anomaly_lower or 'format' in anomaly_lower:
                    suggestions.append("• Review data type validation in ETL processes")
                    suggestions.append("• Check source system data type consistency")
                    suggestions.append("• Verify data transformation logic")
                
                if not suggestions:
                    suggestions = [
                        "• Review data entry and validation processes",
                        "• Check ETL/ELT transformation logic",
                        "• Verify source system data quality",
                        "• Review business rules and constraints"
                    ]
                
                return "Potential root causes:\n" + "\n".join(suggestions)
                
            except Exception as e:
                logger.error(f"Anomaly cause analysis error: {str(e)}")
                return f"Error analyzing anomaly causes: {str(e)}"
        
        return Tool(
            name="analyze_anomaly_causes",
            func=analyze_anomaly_causes,
            description="Analyze potential root causes for detected anomalies and provide remediation suggestions"
        )
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection workflow"""
        try:
            logger.info(f"Starting anomaly detection for table: {input_data.get('table_name')}")
            
            if not self.validate_input(input_data):
                return {
                    'success': False,
                    'error': 'Invalid input data'
                }
            
            table_name = input_data['table_name']
            
            # Use agent executor if we have tools
            if self.agent_executor:
                # Prepare input for agent
                agent_input = {
                    'input': f"Analyze table {table_name} for anomalies and provide detailed report"
                }
                
                # Execute agent
                result = await self.agent_executor.ainvoke(agent_input)
                
                return {
                    'success': True,
                    'table_name': table_name,
                    'analysis': result['output'],
                    'agent_used': True
                }
            else:
                # Fallback to direct analysis
                return await self._direct_anomaly_detection(table_name)
                
        except Exception as e:
            logger.error(f"Anomaly detection agent error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _direct_anomaly_detection(self, table_name: str) -> Dict[str, Any]:
        """Direct anomaly detection without agent tools"""
        try:
            # Get sample data
            sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
            
            if sample_df.empty:
                return {
                    'success': False,
                    'error': f"No data found in table: {table_name}"
                }
            
            # Detect anomalies
            numeric_anomalies = self.data_analyzer.detect_anomalies(sample_df, method='IQR')
            
            # Analyze data quality for other anomaly types
            schema_df = self.snowflake_manager.get_table_schema(table_name)
            quality_issues = self.data_analyzer.analyze_data_quality(sample_df, schema_df)
            
            # Categorize anomalies
            anomalies_by_type = {
                'numeric': [],
                'temporal': [],
                'categorical': [],
                'pattern': [],
                'data_quality': []
            }
            
            for anomaly in numeric_anomalies:
                anomalies_by_type['numeric'].append(anomaly)
            
            for issue in quality_issues:
                if issue['issue_type'] in ['null_values', 'data_type_mismatch']:
                    anomalies_by_type['data_quality'].append(issue)
            
            # Calculate summary
            total_anomalies = sum(len(anomalies) for anomalies in anomalies_by_type.values())
            critical_anomalies = sum(1 for anomalies in anomalies_by_type.values() 
                                   for anomaly in anomalies if anomaly.get('severity') == 'critical')
            
            analysis = {
                'total_anomalies': total_anomalies,
                'critical_anomalies': critical_anomalies,
                'affected_columns': len(set(anomaly.get('column', '') for anomalies in anomalies_by_type.values() 
                                          for anomaly in anomalies if 'column' in anomaly)),
                'anomaly_types': {k: len(v) for k, v in anomalies_by_type.items() if v},
                'anomalies': anomalies_by_type,
                'recommendations': self._generate_anomaly_recommendations(anomalies_by_type)
            }
            
            return {
                'success': True,
                'table_name': table_name,
                'analysis': analysis,
                'agent_used': False
            }
            
        except Exception as e:
            logger.error(f"Direct anomaly detection error: {str(e)}")
            raise
    
    def _generate_anomaly_recommendations(self, anomalies_by_type: Dict[str, List]) -> List[str]:
        """Generate recommendations based on detected anomalies"""
        recommendations = []
        
        # Check for critical anomalies
        critical_anomalies = []
        for anomaly_type, anomalies in anomalies_by_type.items():
            critical_anomalies.extend([a for a in anomalies if a.get('severity') == 'critical'])
        
        if critical_anomalies:
            recommendations.append("Immediate action required for critical anomalies")
        
        # Specific recommendations by type
        if anomalies_by_type.get('numeric'):
            recommendations.append("Review numeric columns for outliers and extreme values")
        
        if anomalies_by_type.get('data_quality'):
            recommendations.append("Address data quality issues (nulls, type mismatches)")
        
        if anomalies_by_type.get('pattern'):
            recommendations.append("Investigate pattern anomalies (duplicates, constant columns)")
        
        # General recommendations
        if recommendations:
            recommendations.append("Implement data validation rules to prevent recurrence")
            recommendations.append("Set up monitoring for similar anomalies")
        else:
            recommendations.append("No major anomalies detected - maintain current data quality practices")
        
        return recommendations
    
    def get_required_input_fields(self) -> list:
        """Get required input fields"""
        return ['table_name']
    
# if __name__ == "__main__":
#     """Test the AnomalyAgent"""
#     import asyncio
#     from dotenv import load_dotenv
    
#     async def test_anomaly_agent():
#         """Test function for anomaly detection agent"""
#         try:
#             # Load environment variables
#             load_dotenv()
            
#             print("=== Testing Anomaly Detection Agent ===")
            
#             # Initialize Snowflake Manager
#             print("\n1. Initializing Snowflake connection...")
            
#             # Based on the error, SnowflakeManager likely loads config from env vars directly
#             # So we don't need to pass any arguments
#             snowflake_manager = SnowflakeManager()
            
#             # Test connection with a simple query
#             try:
#                 test_query = "SELECT 1 as test"
#                 result = snowflake_manager.execute_query(test_query)
#                 print(f"Snowflake connection test: {'Success' if result else 'Failed'}")
                
#                 if not result:
#                     print("Connection failed. Using mock mode...")
#                     # We'll use the actual agent but it will handle failures gracefully
#             except Exception as e:
#                 print(f"Connection error (expected if no Snowflake config): {str(e)[:100]}...")
#                 print("Continuing with agent tools that will handle errors gracefully...")
            
#             # Create Anomaly Agent
#             print("\n2. Creating Anomaly Detection Agent...")
#             agent = AnomalyAgent(snowflake_manager)
#             print(f"Agent created: {agent.name}")
#             print(f"Available tools: {[tool.name for tool in agent.tools]}")
            
#             # Get test table from environment or use default
#             test_table = os.getenv('TEST_TABLE', 'SAMPLE_SALES_DATA')
#             print(f"\n3. Testing with table: {test_table}")
            
#             # Test each tool individually
#             print("\n4. Testing individual tools:")
            
#             # Tool 0: Numeric anomaly detection
#             print("\n   a) Numeric anomaly detection:")
#             try:
#                 numeric_result = agent.tools[0].func(test_table, method="IQR")
#                 print(f"   Result: {numeric_result[:200]}..." if len(numeric_result) > 200 else f"   Result: {numeric_result}")
#             except Exception as e:
#                 print(f"   Error: {str(e)[:100]}...")
            
#             # Tool 1: Temporal anomaly detection
#             print("\n   b) Temporal anomaly detection:")
#             date_column = os.getenv('DATE_COLUMN', 'ORDER_DATE')
#             try:
#                 temporal_result = agent.tools[1].func(test_table, date_column)
#                 print(f"   Result: {temporal_result[:200]}..." if len(temporal_result) > 200 else f"   Result: {temporal_result}")
#             except Exception as e:
#                 print(f"   Error: {str(e)[:100]}...")
            
#             # Tool 2: Categorical anomaly detection
#             print("\n   c) Categorical anomaly detection:")
#             # Try common categorical columns
#             cat_columns = ['CATEGORY', 'STATUS', 'TYPE', 'PRODUCT_NAME']
#             for col in cat_columns:
#                 try:
#                     cat_result = agent.tools[2].func(test_table, col)
#                     if "Error" not in cat_result and "not found" not in cat_result:
#                         print(f"   Testing column '{col}': {cat_result[:150]}...")
#                         break
#                 except:
#                     continue
            
#             # Tool 3: Pattern detection
#             print("\n   d) Pattern anomaly detection:")
#             try:
#                 pattern_result = agent.tools[3].func(test_table)
#                 print(f"   Result: {pattern_result[:200]}..." if len(pattern_result) > 200 else f"   Result: {pattern_result}")
#             except Exception as e:
#                 print(f"   Error: {str(e)[:100]}...")
            
#             # Tool 4: Cause analysis
#             print("\n   e) Anomaly cause analysis:")
#             test_anomaly = "Found 15 future dates in ORDER_DATE column"
#             try:
#                 cause_analysis = agent.tools[4].func(test_table, test_anomaly)
#                 print(f"   Analysis for '{test_anomaly}':\n   {cause_analysis}")
#             except Exception as e:
#                 print(f"   Error: {str(e)[:100]}...")
            
#             # Test 5: Run full agent analysis (if agent_executor is available)
#             print("\n5. Testing full agent run...")
#             input_data = {
#                 'table_name': test_table,
#                 'analysis_type': 'comprehensive'
#             }
            
#             try:
#                 result = await agent.run(input_data)
                
#                 if result['success']:
#                     print(f"   Agent run successful!")
#                     print(f"   Agent used: {result.get('agent_used', False)}")
                    
#                     if 'analysis' in result:
#                         if isinstance(result['analysis'], str):
#                             print(f"   Analysis (first 300 chars):\n   {result['analysis'][:300]}...")
#                         elif isinstance(result['analysis'], dict):
#                             print(f"   Analysis summary:")
#                             for key, value in list(result['analysis'].items())[:5]:  # Show first 5 items
#                                 if isinstance(value, (list, dict)) and len(str(value)) > 100:
#                                     print(f"     {key}: {type(value).__name__} ({len(value) if isinstance(value, list) else len(value)} items)")
#                                 else:
#                                     print(f"     {key}: {value}")
#                 else:
#                     print(f"   Agent run failed: {result.get('error', 'Unknown error')}")
#             except Exception as e:
#                 print(f"   Error running agent: {str(e)[:100]}...")
            
#             print("\n=== Test Complete ===")
            
#         except Exception as e:
#             print(f"Error during testing: {str(e)}")
#             import traceback
#             traceback.print_exc()
    
#     # Run the test
#     asyncio.run(test_anomaly_agent())