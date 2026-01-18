"""
Data profiling agent using LangChain
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger
from langchain_core.tools import Tool
from langchain.tools import tool
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeManager
from tools.data_tools import DataAnalyzer
from config.prompts import PROFILER_AGENT_PROMPT

class ProfilerAgent(BaseAgent):
    """Agent for data profiling tasks"""
    
    def __init__(self, snowflake_manager: SnowflakeManager):
        """Initialize profiler agent"""
        self.snowflake_manager = snowflake_manager
        self.data_analyzer = DataAnalyzer()
        
        # Define agent tools
        tools = [
            self.get_table_schema_tool(),
            self.get_table_stats_tool(),
            self.analyze_data_quality_tool(),
            self.check_constraints_tool()
        ]
        
        super().__init__(
            name="Data Profiler Agent",
            description="Profiles database tables and analyzes data quality",
            tools=tools
        )
    
    def get_table_schema_tool(self) -> Tool:
        """Tool to get table schema"""
        def get_table_schema(table_name: str) -> str:
            """Get the schema of a table including column names and data types"""
            try:
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                if schema_df.empty:
                    return f"No schema found for table: {table_name}"
                
                schema_info = []
                for _, row in schema_df.iterrows():
                    schema_info.append(
                        f"Column: {row['COLUMN_NAME']}, "
                        f"Type: {row['DATA_TYPE']}, "
                        f"Nullable: {row['IS_NULLABLE']}"
                    )
                
                return "\n".join(schema_info)
                
            except Exception as e:
                logger.error(f"Schema tool error: {str(e)}")
                return f"Error getting schema: {str(e)}"
        
        return Tool(
            name="get_table_schema",
            description="Get table schema including columns and data types",
            func=get_table_schema
        )
    
    def get_table_stats_tool(self) -> Tool:
        """Tool to get table statistics"""
        def get_table_stats(table_name: str) -> str:
            """Get statistical information about a table"""
            try:
                # Get row count
                row_count = self.snowflake_manager.get_row_count(table_name)
                
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=1000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Calculate basic statistics
                stats = {
                    'table_name': table_name,
                    'row_count': row_count,
                    'column_count': len(sample_df.columns),
                    'memory_usage_mb': round(sample_df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
                
                # Add column-level stats
                column_stats = []
                for column in sample_df.columns:
                    col_data = sample_df[column]
                    col_stats = {
                        'column': column,
                        'dtype': str(col_data.dtype),
                        'null_count': int(col_data.isnull().sum()),
                        'null_percentage': round(col_data.isnull().sum() / len(col_data) * 100, 2),
                        'unique_count': int(col_data.nunique()),
                        'unique_percentage': round(col_data.nunique() / len(col_data) * 100, 2)
                    }
                    
                    # Add numeric stats if applicable
                    if pd.api.types.is_numeric_dtype(col_data):
                        numeric_data = col_data.dropna()
                        if len(numeric_data) > 0:
                            col_stats.update({
                                'min': float(numeric_data.min()),
                                'max': float(numeric_data.max()),
                                'mean': float(numeric_data.mean()),
                                'std': float(numeric_data.std()),
                                'median': float(numeric_data.median())
                            })
                    
                    column_stats.append(col_stats)
                
                stats['column_stats'] = column_stats
                
                # Format for readability
                result = []
                result.append(f"Table: {stats['table_name']}")
                result.append(f"Row Count: {stats['row_count']:,}")
                result.append(f"Column Count: {stats['column_count']}")
                result.append(f"Estimated Memory Usage: {stats['memory_usage_mb']} MB")
                result.append("\nColumn Statistics:")
                
                for col_stat in stats['column_stats']:
                    result.append(f"\n  Column: {col_stat['column']}")
                    result.append(f"    Type: {col_stat['dtype']}")
                    result.append(f"    Nulls: {col_stat['null_count']} ({col_stat['null_percentage']}%)")
                    result.append(f"    Unique: {col_stat['unique_count']} ({col_stat['unique_percentage']}%)")
                    
                    if 'min' in col_stat:
                        result.append(f"    Min: {col_stat['min']:.2f}")
                        result.append(f"    Max: {col_stat['max']:.2f}")
                        result.append(f"    Mean: {col_stat['mean']:.2f}")
                        result.append(f"    Std Dev: {col_stat['std']:.2f}")
                
                return "\n".join(result)
                
            except Exception as e:
                logger.error(f"Stats tool error: {str(e)}")
                return f"Error getting statistics: {str(e)}"
        
        return Tool(
            name="get_table_stats",
            description="Get statistical information about a table",
            func=get_table_stats
        )
    
    def analyze_data_quality_tool(self) -> Tool:
        """Tool to analyze data quality"""
        def analyze_data_quality(table_name: str) -> str:
            """Analyze data quality issues in a table"""
            try:
                # Get schema
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
                
                if sample_df.empty:
                    return f"No data to analyze in table: {table_name}"
                
                # Analyze data quality
                quality_issues = self.data_analyzer.analyze_data_quality(sample_df, schema_df)
                
                # Format results
                if not quality_issues:
                    return f"No data quality issues found in table: {table_name}"
                
                issues_text = []
                issues_text.append(f"Data Quality Issues for table: {table_name}")
                issues_text.append("=" * 50)
                
                for issue in quality_issues:
                    issues_text.append(
                        f"\nColumn: {issue.get('column', 'Unknown')}\n"
                        f"  Issue Type: {issue.get('issue_type', 'Unknown')}\n"
                        f"  Count: {issue.get('count', 0)}\n"
                        f"  Severity: {issue.get('severity', 'Medium')}\n"
                        f"  Details: {issue.get('details', 'No additional details')}"
                    )
                
                return "\n".join(issues_text)
                
            except Exception as e:
                logger.error(f"Data quality tool error: {str(e)}")
                return f"Error analyzing data quality: {str(e)}"
        
        return Tool(
            name="analyze_data_quality",
            description="Analyze data quality issues in a table",
            func=analyze_data_quality
        )
    
    def check_constraints_tool(self) -> Tool:
        """Tool to check constraints"""
        def check_constraints(table_name: str) -> str:
            """Check if data violates given constraints"""
            try:
                # Default constraints
                constraints = {
                    'not_null': ['id', 'created_at'],
                    'unique': ['email', 'username'],
                    'range_checks': {
                        'age': {'min': 0, 'max': 150},
                        'salary': {'min': 0}
                    }
                }
                
                # Get sample data
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data to check in table: {table_name}"
                
                # Check constraints
                violations = self.data_analyzer.check_constraints(sample_df, constraints)
                
                if not violations:
                    return f"No constraint violations found in table: {table_name}"
                
                violations_text = []
                violations_text.append(f"Constraint Violations for table: {table_name}")
                violations_text.append("=" * 50)
                
                for violation in violations:
                    violations_text.append(
                        f"\nConstraint: {violation.get('constraint', 'Unknown')}\n"
                        f"  Column: {violation.get('column', 'Unknown')}\n"
                        f"  Violation Count: {violation.get('count', 0)}\n"
                        f"  Sample Values: {violation.get('sample_values', [])[:3]}\n"
                        f"  Details: {violation.get('details', 'No additional details')}"
                    )
                
                return "\n".join(violations_text)
                
            except Exception as e:
                logger.error(f"Constraint tool error: {str(e)}")
                return f"Error checking constraints: {str(e)}"
        
        return Tool(
            name="check_constraints",
            description="Check if data violates given constraints",
            func=check_constraints
        )
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute profiling workflow"""
        try:
            logger.info(f"Starting profiling for table: {input_data.get('table_name')}")
            
            if not self.validate_input(input_data):
                return {
                    'success': False,
                    'error': 'Invalid input data. Required: table_name'
                }
            
            table_name = input_data['table_name']
            
            # Use agent executor if we have tools
            if self.agent_executor:
                # Prepare input for agent
                agent_input = {
                    'input': f"Profile the table '{table_name}' and provide comprehensive analysis including: "
                             f"1. Table schema, 2. Basic statistics, 3. Data quality assessment, "
                             f"4. Any constraint violations. Use all available tools to gather this information."
                }
                
                # Execute agent
                result = await self.agent_executor.ainvoke(agent_input)
                
                return {
                    'success': True,
                    'table_name': table_name,
                    'analysis': result.get('output', 'No output generated'),
                    'agent_used': True,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            else:
                # Fallback to direct analysis
                return await self._direct_profiling(table_name)
                
        except Exception as e:
            logger.error(f"Profiling agent error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'table_name': input_data.get('table_name', 'Unknown')
            }
    
    async def _direct_profiling(self, table_name: str) -> Dict[str, Any]:
        """Direct profiling without agent tools"""
        try:
            # Get schema
            schema_df = self.snowflake_manager.get_table_schema(table_name)
            
            # Get statistics
            sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
            
            if sample_df.empty:
                return {
                    'success': False,
                    'error': f"No data found in table: {table_name}"
                }
            
            # Analyze data
            analysis = self.data_analyzer.comprehensive_analysis(sample_df, schema_df)
            
            return {
                'success': True,
                'table_name': table_name,
                'analysis': analysis,
                'agent_used': False,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Direct profiling error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_required_input_fields(self) -> list:
        """Get required input fields"""
        return ['table_name']