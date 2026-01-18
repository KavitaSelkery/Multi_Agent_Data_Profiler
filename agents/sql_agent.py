"""
SQL generation and execution agent
"""
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from loguru import logger
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeManager
from config.prompts import SQL_AGENT_PROMPT
from config.settings import Settings

class SQLAgent(BaseAgent):
    """Agent for SQL generation and execution"""
    
    def __init__(self, snowflake_manager: SnowflakeManager):
        """Initialize SQL agent"""
        self.snowflake_manager = snowflake_manager
        self.settings = Settings()
        
        # Define agent tools
        tools = [
            self.generate_sql_tool(),
            self.validate_sql_tool(),
            self.execute_sql_tool(),
            self.explain_sql_tool(),
            self.optimize_sql_tool()
        ]
        
        super().__init__(
            name="SQL Generation Agent",
            description="Generates, validates, and executes SQL queries from natural language",
            tools=tools
        )
        
        # Initialize SQL-specific LLM
        self.sql_llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            max_tokens=self.settings.AGENT_MAX_TOKENS,
            api_key=self.settings.OPENAI_API_KEY
        )
    
    def generate_sql_tool(self) -> Tool:
        """Tool to generate SQL from natural language"""
        def generate_sql(query: str) -> str:
            """Generate SQL query from natural language description"""
            try:
                # Get database context
                context = self._get_database_context()
                
                # Create prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SQL_AGENT_PROMPT + "\n\n" + context),
                    ("human", "Generate a SQL query for: {query}")
                ])
                
                # Create chain
                chain = prompt | self.sql_llm
                
                # Generate SQL
                result = chain.invoke({"query": query})
                
                # Extract SQL
                sql_query = result.content.strip()
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                # Ensure it's valid
                if not sql_query.upper().startswith(('SELECT', 'WITH', 'DESC', 'SHOW', 'EXPLAIN')):
                    sql_query = f"SELECT {sql_query}"
                
                return f"Generated SQL:\n```sql\n{sql_query}\n```"
                
            except Exception as e:
                logger.error(f"SQL generation error: {str(e)}")
                return f"Error generating SQL: {str(e)}"
        
        return Tool(
            name="generate_sql",
            description="Generate SQL query from natural language description",
            func=generate_sql
        )
    
    def validate_sql_tool(self) -> Tool:
        """Tool to validate SQL syntax"""
        def validate_sql(sql_query: str) -> str:
            """Validate SQL query syntax and compatibility with Snowflake"""
            try:
                is_valid, message = self.snowflake_manager.validate_query(sql_query)
                
                if is_valid:
                    return f"✅ SQL query is valid:\n```sql\n{sql_query}\n```"
                else:
                    suggestions = self._suggest_sql_fixes(sql_query, message)
                    return f"❌ SQL validation failed:\nError: {message}\nSuggestions:\n{suggestions}"
                    
            except Exception as e:
                logger.error(f"SQL validation error: {str(e)}")
                return f"Error validating SQL: {str(e)}"
        
        return Tool(
            name="validate_sql",
            description="Validate SQL query syntax and Snowflake compatibility",
            func=validate_sql
        )
    
    def execute_sql_tool(self) -> Tool:
        """Tool to execute SQL queries"""
        def execute_sql(sql_query: str) -> str:
            """Execute SQL query and return results"""
            try:
                # Add LIMIT if not present and it's a SELECT query
                if sql_query.strip().upper().startswith('SELECT') and 'LIMIT' not in sql_query.upper():
                    sql_query = f"{sql_query.rstrip(';')} LIMIT 100"
                
                # Execute query
                df = self.snowflake_manager.execute_query(sql_query)
                
                # Check for errors
                if not isinstance(df, pd.DataFrame):
                    return f"❌ Query execution failed: Unexpected result type"
                
                if df.empty:
                    return "✅ Query executed successfully but returned no results."
                
                # Format results
                row_count = len(df)
                column_count = len(df.columns)
                
                # Get sample of results
                sample_size = min(10, row_count)
                sample_df = df.head(sample_size)
                
                # Create summary
                summary = f"""
                ✅ Query executed successfully!
                
                Results:
                - Rows: {row_count:,}
                - Columns: {column_count}
                
                Sample Results (first {sample_size} rows):
                ```
                {sample_df.to_string(index=False, max_rows=sample_size, max_cols=min(10, column_count))}
                ```
                
                Query:
                ```sql
                {sql_query}
                ```
                """
                
                return summary
                
            except Exception as e:
                logger.error(f"SQL execution error: {str(e)}")
                return f"Error executing SQL: {str(e)}"
        
        return Tool(
            name="execute_sql",
            description="Execute SQL query and return formatted results",
            func=execute_sql
        )
    
    def explain_sql_tool(self) -> Tool:
        """Tool to explain SQL query execution"""
        def explain_sql(sql_query: str) -> str:
            """Explain what a SQL query does in plain language"""
            try:
                # Use LLM to explain the query
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a SQL expert. Explain what the following SQL query does in simple, clear terms."),
                    ("human", "Explain this SQL query: {sql_query}")
                ])
                
                chain = prompt | self.sql_llm
                explanation = chain.invoke({"sql_query": sql_query})
                
                return f"""
                Query Explanation:
                
                {explanation.content}
                
                Original Query:
                ```sql
                {sql_query}
                ```
                """
                
            except Exception as e:
                logger.error(f"SQL explanation error: {str(e)}")
                return f"Error explaining SQL: {str(e)}"
        
        return Tool(
            name="explain_sql",
            description="Explain what a SQL query does in plain language",
            func=explain_sql
        )
    
    def optimize_sql_tool(self) -> Tool:
        """Tool to optimize SQL queries"""
        def optimize_sql(sql_query: str) -> str:
            """Optimize SQL query for better performance"""
            try:
                # Use LLM to suggest optimizations
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a SQL performance expert. Analyze the following SQL query and suggest optimizations for Snowflake."""),
                    ("human", "Optimize this SQL query for Snowflake: {sql_query}")
                ])
                
                chain = prompt | self.sql_llm
                optimization = chain.invoke({"sql_query": sql_query})
                
                return f"""
                Optimization Suggestions:
                
                {optimization.content}
                
                Original Query:
                ```sql
                {sql_query}
                ```
                """
                
            except Exception as e:
                logger.error(f"SQL optimization error: {str(e)}")
                return f"Error optimizing SQL: {str(e)}"
        
        return Tool(
            name="optimize_sql",
            description="Optimize SQL query for better performance in Snowflake",
            func=optimize_sql
        )
    
    def _get_database_context(self) -> str:
        """Get database context for SQL generation"""
        try:
            context_parts = []
            
            # Get current database and schema
            current_db = self.snowflake_manager.execute_query("SELECT CURRENT_DATABASE()")
            current_schema = self.snowflake_manager.execute_query("SELECT CURRENT_SCHEMA()")
            
            if not current_db.empty:
                context_parts.append(f"Current Database: {current_db.iloc[0, 0]}")
            if not current_schema.empty:
                context_parts.append(f"Current Schema: {current_schema.iloc[0, 0]}")
            
            # Get tables
            tables = self.snowflake_manager.get_tables()
            if tables:
                context_parts.append(f"Available Tables: {', '.join(tables[:10])}")
                if len(tables) > 10:
                    context_parts.append(f"... and {len(tables) - 10} more tables")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Database context error: {str(e)}")
            return "Database context unavailable"
    
    def _suggest_sql_fixes(self, sql_query: str, error_message: str) -> str:
        """Suggest fixes for SQL errors"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a SQL debugging expert. Given a SQL query and error message, suggest specific fixes."""),
                ("human", "SQL Query:\n```sql\n{sql_query}\n```\n\nError:\n{error_message}")
            ])
            
            chain = prompt | self.sql_llm
            suggestions = chain.invoke({
                "sql_query": sql_query,
                "error_message": error_message
            })
            
            return suggestions.content
            
        except Exception as e:
            logger.error(f"SQL fix suggestions error: {str(e)}")
            return "Unable to generate specific fixes."
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL generation workflow"""
        try:
            logger.info(f"Starting SQL generation for query: {input_data.get('query', '')[:50]}...")
            
            if not self.validate_input(input_data):
                return {
                    'success': False,
                    'error': 'Invalid input data'
                }
            
            query = input_data['query']
            
            # Use agent executor if we have tools
            if self.agent_executor:
                # Prepare input for agent
                agent_input = {
                    'input': f"Generate and execute SQL for: {query}"
                }
                
                # Execute agent
                result = await self.agent_executor.ainvoke(agent_input)
                
                return {
                    'success': True,
                    'agent_response': result.get('output', 'No output'),
                    'agent_used': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback to direct SQL generation
                return await self._direct_sql_generation(query)
                
        except Exception as e:
            logger.error(f"SQL agent error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _direct_sql_generation(self, query: str) -> Dict[str, Any]:
        """Direct SQL generation without agent tools"""
        try:
            # Generate SQL using LLM
            context = self._get_database_context()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", SQL_AGENT_PROMPT + "\n\n" + context),
                ("human", "Generate SQL for: {query}")
            ])
            
            chain = prompt | self.sql_llm
            result = await chain.ainvoke({"query": query})
            
            # Extract SQL
            sql_query = result.content.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Add LIMIT if not present
            if sql_query.upper().startswith('SELECT') and 'LIMIT' not in sql_query.upper():
                sql_query = f"{sql_query} LIMIT 100"
            
            # Execute SQL
            df = self.snowflake_manager.execute_query(sql_query)
            
            # Prepare result
            result_summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'execution_success': True
            }
            
            return {
                'success': True,
                'generated_sql': sql_query,
                'result_summary': result_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Direct SQL generation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_required_input_fields(self) -> list:
        """Get required input fields"""
        return ['query']